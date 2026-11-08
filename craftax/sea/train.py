"""Command-line entry point for the Craftax-Classic SEA PPO-RNN pipeline."""

from __future__ import annotations

import argparse
import json
import math
import time
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from flax import serialization

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from craftax.sea.checkpoint import save_params
from craftax.sea.discovery import (
    DiscoveryConfig,
    embed_positive_transitions,
    fit_clusters,
    summarize_cluster_achievements,
)
from craftax.sea.dynamic_graph import DynamicGraphConfig
from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime, load_oracle_graph
from craftax.sea.metrics import (
    EASY17_ACHIEVEMENT_INDICES,
    NO_EAT_PLANT_ACHIEVEMENT_INDICES,
    summarize_training_metrics,
)
from craftax.sea.ppo_rnn import PPOConfig, make_train
from craftax.sea.rollout import iter_policy_transitions
from craftax.sea.streaming_discovery import collect_clustering_dataset, train_streaming_encoder


def _parse_bool(value):
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


class _WandbLogger:
    """Optional host-side W&B logging for JAX debug callbacks."""

    def __init__(self, args, config):
        self.run = None
        if args.wandb_project is None:
            return
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "W&B logging requires the 'wandb' package; install it with "
                "`pip install wandb` or install the sea extra"
            ) from error

        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name, group=args.wandb_group, mode=args.wandb_mode,
            tags=args.wandb_tags, config=config, dir=str(Path(args.output).resolve()),
        )
        for stage in ("base", "goal"):
            self.run.define_metric(f"{stage}/steps")
            self.run.define_metric(f"{stage}/*", step_metric=f"{stage}/steps")
        self.run.define_metric("encoder/interactions")
        self.run.define_metric("encoder/*", step_metric="encoder/interactions")

    @staticmethod
    def _scalars(metrics):
        values = {}
        for name, value in metrics.items():
            array = np.asarray(value)
            if array.ndim == 0:
                values[name] = float(array)
            elif name == "selected_goal_distribution" and array.ndim == 1:
                values.update(
                    {
                        f"selected_goal_probability_{index}": float(probability)
                        for index, probability in enumerate(array)
                    }
                )
        return values

    def log_policy(self, stage, steps, metrics, steps_per_second=None):
        if self.run is None:
            return
        values = {f"{stage}/steps": steps, **{
            f"{stage}/{name}": value
            for name, value in self._scalars(metrics).items()
        }}
        if steps_per_second is not None:
            values[f"{stage}/steps_per_second"] = steps_per_second
        self.run.log(values)

    def log_encoder(self, update, metrics, counters):
        if self.run is None:
            return
        self.run.log({
            "encoder/interactions": counters["prediction_transitions_seen"],
            "encoder/update": update,
            "encoder/total_loss": float(metrics[0]),
            "encoder/prediction_loss": float(metrics[1][0]),
            "encoder/contrast_loss": float(metrics[1][1]),
            "encoder/gradient_norm": float(metrics[2]),
            "encoder/positive_fraction": counters["prediction_positive_fraction"],
            "encoder/contrast_episodes": counters["contrast_episodes"],
        })

    def finish(self):
        if self.run is not None:
            self.run.finish()


def _save_metric_summary(path, metrics):
    np.savez_compressed(
        path.with_name(f"{path.stem}_history.npz"),
        **{name: np.asarray(value) for name, value in metrics.items()},
    )
    summary = summarize_training_metrics(metrics)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _progress_reporter(output, stage, config, wandb_logger=None):
    started = time.perf_counter()
    previous = [None, None]
    total_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    progress = tqdm(
        total=total_updates,
        desc=f"[{stage}] PPO",
        unit="update",
        dynamic_ncols=True,
    )

    def report(update, metrics, params):
        update = int(update)
        progress.update(max(0, update - progress.n))
        steps = update * config.num_envs * config.num_steps
        now = time.perf_counter()
        record = {
            "stage": stage,
            "update": update,
            "steps": steps,
            "total_updates": config.total_timesteps // (config.num_envs * config.num_steps),
            "elapsed_seconds": now - started,
            "last_update": {name: np.asarray(value).tolist() for name, value in metrics.items()},
        }
        if previous[0] is not None:
            record["steps_per_second"] = (steps - previous[0]) / (now - previous[1])
        previous[:] = [steps, now]
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise FloatingPointError(f"non-finite {stage} metrics at update {update}")
        checkpoint_params = params
        q_checkpoint_params = None
        if isinstance(params, dict) and "policy" in params:
            checkpoint_params = params["policy"]
            q_checkpoint_params = {
                name: value for name, value in params.items() if name != "policy"
            }
        checkpoint = output / f"{stage}_policy_latest.msgpack"
        temporary = checkpoint.with_suffix(".tmp")
        save_params(temporary, checkpoint_params)
        temporary.replace(checkpoint)
        if q_checkpoint_params is not None:
            q_checkpoint = output / "high_level_q_latest.msgpack"
            q_temporary = q_checkpoint.with_suffix(".tmp")
            save_params(q_temporary, q_checkpoint_params)
            q_temporary.replace(q_checkpoint)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        status = output / "progress.tmp"
        status.write_text(json.dumps(record, indent=2), encoding="utf-8")
        status.replace(output / "progress.json")
        if wandb_logger is not None:
            wandb_logger.log_policy(stage, steps, metrics, record.get("steps_per_second"))
        progress.set_postfix(
            loss=f"{float(metrics['loss']):.4f}",
            return_=f"{float(metrics['mean_episode_return']):.2f}",
            sps=f"{record.get('steps_per_second', 0):.0f}",
        )
        tqdm.write(
            f"[{stage}] update={update}/{record['total_updates']} steps={steps} "
            f"loss={float(metrics['loss']):.6f} "
            f"return={float(metrics['mean_episode_return']):.3f} "
            f"sps={record.get('steps_per_second', 0):.1f}"
        )
        if update >= total_updates:
            progress.close()

    return report


def _make_dynamic_config(args):
    if args.pixels or args.high_level_controller != "residual":
        if args.dynamic_lambda != 1.0 or args.dynamic_score_source != "residual":
            raise ValueError("dynamic scoring options require --symbolic --high-level-controller residual")
        return None
    config = DynamicGraphConfig(
        warmup_steps=args.dynamic_warmup_steps,
        residual_weight=args.dynamic_beta,
        structural_prior_weight=args.dynamic_lambda,
        score_source=args.dynamic_score_source,
        temperature=args.dynamic_temperature,
        replay_capacity=args.high_level_replay_capacity,
        batch_size=args.high_level_q_batch_size,
        learning_rate=args.high_level_q_learning_rate,
        high_level_q_replay_ratio=args.high_level_q_replay_ratio,
        min_replay_size=args.high_level_min_replay_size,
        goal_timeout=args.high_level_goal_timeout,
        allow_unready_goals=args.allow_unready_goals,
        allow_completed_goals=args.allow_completed_goals,
    )
    config.validate()
    return config


def run_pipeline(args):
    train_policy = make_train
    collect_stream = iter_policy_transitions
    achievement_indices = {
        "all": None,
        "no_eat_plant": NO_EAT_PLANT_ACHIEVEMENT_INDICES,
        "easy17": EASY17_ACHIEVEMENT_INDICES,
    }[args.discovery_achievement_subset]
    oracle_source = (
        Path(args.oracle_graph).expanduser().resolve()
        if args.oracle_graph is not None
        else None
    )
    oracle_graph = (
        load_oracle_graph(oracle_source) if oracle_source is not None else None
    )
    if args.allow_unready_goals or args.allow_completed_goals:
        if args.pixels or args.high_level_controller != "residual":
            raise ValueError("goal eligibility options require --symbolic --high-level-controller residual")
    if args.allow_completed_goals and oracle_graph is None:
        raise ValueError("--allow-completed-goals true requires an achievement-indexed --graph-path")
    dynamic_config = _make_dynamic_config(args)
    if args.discovery_interactions <= 0:
        raise ValueError("--discovery-interactions must be positive")
    if args.contrast_interactions < 0:
        raise ValueError("--contrast-interactions cannot be negative")
    if args.contrast_weight < 0:
        raise ValueError("--contrast-weight cannot be negative")
    if args.discovery_dataset_checkpoint:
        raise ValueError("streaming discovery cannot train from an old dataset; use --base-policy-checkpoint")
    if args.discovery_num_envs < 1 or args.discovery_batch_size < 1:
        raise ValueError("discovery environment and batch counts must be positive")
    if args.discovery_batch_size % args.discovery_num_envs:
        raise ValueError("--discovery-batch-size must be divisible by --discovery-num-envs")
    expected_updates = math.ceil(args.discovery_interactions / args.discovery_batch_size)
    if args.discovery_train_steps is not None and args.discovery_train_steps != expected_updates:
        raise ValueError(f"fresh-data training requires {expected_updates} updates for this budget; "
                         "set --discovery-interactions rather than repeating or dropping batches")
    if args.clustering_interactions < 1 or args.clustering_transitions < 2:
        raise ValueError("clustering requires a positive rollout budget and at least two transitions")
    output = Path(args.output)
    if (output / "config.json").exists():
        raise FileExistsError(f"run already exists: {output}; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    if oracle_source is not None:
        shutil.copy2(oracle_source, output / "oracle_graph.npz")
        args.oracle_graph = "oracle_graph.npz"
    symbolic = not args.pixels
    env = make_sea_craftax_classic_env(
        symbolic=symbolic,
        idle_timeout=args.idle_timeout,
        immortal=args.immortal,
        health_reward_coef=args.health_reward_coef,
        combat_damage_mode=args.combat_damage_mode,
        track_achievement_events=args.allow_completed_goals,
    )
    common_ppo = dict(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        max_grad_norm=args.max_grad_norm,
        hidden_size=args.hidden_size,
        reset_ratio=args.reset_ratio,
    )
    base_config = PPOConfig(total_timesteps=args.base_timesteps, **common_ppo)
    discovery_train_steps = expected_updates
    stream_config = replace(
        base_config, num_envs=args.discovery_num_envs,
        num_steps=args.discovery_batch_size // args.discovery_num_envs,
        reset_ratio=math.gcd(args.discovery_num_envs, args.reset_ratio),
        total_timesteps=args.discovery_interactions,
    )
    discovery_config = DiscoveryConfig(
        train_steps=discovery_train_steps,
        batch_size=args.discovery_batch_size,
        hidden_size=args.discovery_hidden_size,
        embedding_size=args.embedding_size,
        learning_rate=args.discovery_learning_rate,
        contrast_weight=args.contrast_weight,
        contrast_groups=args.contrast_groups,
        contrast_episode_capacity=args.contrast_episode_capacity,
        max_grad_norm=args.discovery_max_grad_norm,
    )
    goal_config = PPOConfig(total_timesteps=args.goal_timesteps, **common_ppo)
    run_config = {
        "args": vars(args),
        "base_ppo": asdict(base_config),
        "discovery": asdict(discovery_config),
        "discovery_rollout": asdict(stream_config),
        "discovery_mean_contrast_coefficient": discovery_config.mean_contrast_coefficient,
        "goal_ppo": asdict(goal_config),
        "dynamic_graph": (
            None if dynamic_config is None else asdict(dynamic_config)
        ),
        "oracle_direct": oracle_graph is not None,
        "achievement_reward_policy": "episode-first-occurrence-only",
    }
    (output / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    wandb_logger = _WandbLogger(args, run_config)

    started = time.perf_counter()
    if oracle_graph is not None:
        print(
            "[oracle-direct] skipping base PPO, encoder discovery, and clustering",
            flush=True,
        )
        goal_runtime = GoalRuntime(
            transition_network=None,
            transition_params=None,
            centroids=jnp.zeros(
                (len(oracle_graph.achievement_ids), 1), dtype=jnp.float32
            ),
            threshold=0.0,
            graph=jnp.asarray(oracle_graph.graph),
            exploration_reward_coefficient=args.exploration_reward_coefficient,
            goal_success_coefficient=args.goal_success_coefficient,
            non_target_achievement_coefficient=(
                args.non_target_achievement_coefficient
            ),
            include_new_tasks=not args.no_new_tasks,
            oracle_achievement_ids=oracle_graph.achievement_ids,
            allow_completed_goals=args.allow_completed_goals,
        )
        print(
            "[5/5] training oracle-graph-conditioned PPO-RNN; "
            f"controller={args.high_level_controller}, "
            f"objectives={goal_runtime.num_objectives}",
            flush=True,
        )
        goal_result = jax.jit(
            train_policy(
                goal_config,
                env,
                pixel=args.pixels,
                goal_runtime=goal_runtime,
                dynamic_config=dynamic_config,
                progress_callback=_progress_reporter(
                    output, "goal", goal_config, wandb_logger
                ),
                progress_interval=args.log_interval,
            )
        )(jax.random.PRNGKey(args.seed + 3))
        jax.block_until_ready(goal_result["metrics"]["loss"])
        jax.effects_barrier()
        save_params(
            output / "goal_policy.msgpack",
            goal_result["runner_state"][0].params,
        )
        if dynamic_config is not None:
            dynamic_state = goal_result["runner_state"][7]
            save_params(
                output / "high_level_q.msgpack",
                {
                    "full_q": dynamic_state.q_state.full_q.params,
                    "base_q": dynamic_state.q_state.base_q.params,
                    "target_full_q": dynamic_state.q_state.target_full_params,
                },
            )
        goal_metrics = _save_metric_summary(
            output / "goal_metrics.json", goal_result["metrics"]
        )
        print(goal_metrics)
        wandb_logger.finish()
        print(
            f"oracle-direct pipeline finished in "
            f"{time.perf_counter() - started:.1f}s"
        )
        return

    if args.base_policy_checkpoint:
        checkpoint = Path(args.base_policy_checkpoint).resolve()
        print(f"[1/5] reusing frozen collection policy: {checkpoint}")
        base_params = serialization.msgpack_restore(checkpoint.read_bytes())
        save_params(output / "base_policy.msgpack", base_params)
        source_metrics = checkpoint.parent / "base_metrics.json"
        if source_metrics.exists():
            shutil.copy2(source_metrics, output / "base_metrics.json")
    else:
        print("[1/5] training frozen collection policy")
        base_result = jax.jit(train_policy(
            base_config, env, pixel=args.pixels,
            progress_callback=_progress_reporter(output, "base", base_config, wandb_logger),
            progress_interval=args.log_interval,
        ))(
            jax.random.PRNGKey(args.seed)
        )
        jax.block_until_ready(base_result["metrics"]["loss"])
        jax.effects_barrier()
        base_params = base_result["runner_state"][0].params
        save_params(output / "base_policy.msgpack", base_params)
        base_metrics = _save_metric_summary(
            output / "base_metrics.json", base_result["metrics"]
        )
        print(base_metrics)

    if args.base_only:
        wandb_logger.finish()
        print("base-only pipeline complete", flush=True)
        return

    def encoder_progress(update, metrics, counters):
        record = {"update": update, "total_updates": discovery_config.train_steps,
                  "prediction": float(metrics[1][0]), "contrast": float(metrics[1][1]),
                  "loss": float(metrics[0]), "gradient_norm": float(metrics[2]), **counters}
        if not all(np.isfinite(record[key]) for key in ("prediction", "contrast", "loss", "gradient_norm")):
            raise FloatingPointError(f"non-finite encoder metrics at update {update}")
        with (output / "encoder_progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        wandb_logger.log_encoder(update, metrics, counters)
        tqdm.write(f"[encoder] {record}")

    print("[2-3/5] collecting fresh transitions and training SEA encoder")
    encoder_bar = tqdm(
        total=discovery_config.train_steps,
        desc="[2-3/5] SEA encoder",
        unit="update",
        dynamic_ncols=True,
    )

    def encoder_step(update, interactions):
        encoder_bar.update(max(0, update - encoder_bar.n))
        encoder_bar.set_postfix(interactions=f"{interactions / 1e6:.2f}M")
    stream = collect_stream(
        env, base_params, stream_config, interactions=args.discovery_interactions,
        pixel=args.pixels, seed=args.seed + 1, chunk_steps=stream_config.num_steps,
    )
    result = train_streaming_encoder(
        stream, discovery_config, contrast_interactions=args.contrast_interactions,
        pixel=args.pixels, seed=args.seed + 2, progress_callback=encoder_progress,
        achievement_indices=achievement_indices, step_callback=encoder_step,
    )
    encoder_bar.close()
    if result.summary["prediction_transitions_seen"] != args.discovery_interactions:
        raise RuntimeError("encoder did not consume the full fresh-transition budget")
    result.summary["achievement_subset"] = args.discovery_achievement_subset
    result.summary["achievement_indices"] = (
        None if achievement_indices is None else list(achievement_indices)
    )
    (output / "discovery_summary.json").write_text(json.dumps(result.summary, indent=2))
    if result.contrast_dataset is not None:
        result.contrast_dataset.save(output / "contrast_episode_dataset.npz")
    encoder, encoder_state, encoder_metrics = result.network, result.state, result.metrics
    save_params(output / "transition_encoder.msgpack", encoder_state.params)
    print(
        {
            "total": float(encoder_metrics[0]),
            "prediction": float(encoder_metrics[1][0]),
            "contrast": float(encoder_metrics[1][1]),
            "gradient_norm": float(encoder_metrics[2]),
            "gradient_was_clipped": bool(
                encoder_metrics[2] > discovery_config.max_grad_norm
            ),
        }
    )

    print("[4/5] collecting independent clustering transitions and fitting clusters")
    clustering_stream = collect_stream(
        env, base_params, stream_config, interactions=args.clustering_interactions,
        pixel=args.pixels, seed=args.seed + 4, chunk_steps=args.collect_chunk_steps,
    )
    clustering_bar = tqdm(
        total=args.clustering_interactions,
        desc="[4/5] clustering collection",
        unit="interaction",
        unit_scale=True,
        dynamic_ncols=True,
    )

    def clustering_progress(consumed, positives):
        clustering_bar.update(max(0, consumed - clustering_bar.n))
        clustering_bar.set_postfix(
            positives=f"{positives}/{args.clustering_transitions}"
        )

    dataset, clustering_consumed = collect_clustering_dataset(
        clustering_stream, max_transitions=args.clustering_transitions,
        achievement_indices=achievement_indices,
        progress_callback=clustering_progress,
    )
    clustering_bar.close()
    clustering_stream.close()
    dataset.save(output / "clustering_dataset.npz")
    (output / "clustering_collection.json").write_text(json.dumps({
        "seed": args.seed + 4, "rollout_interactions": clustering_consumed,
        "positive_transitions": len(dataset), "independent_of_encoder_training": True,
    }, indent=2))
    embeddings, episode_id, episode_step = embed_positive_transitions(
        encoder, encoder_state.params, dataset, max_transitions=None
    )
    artifacts = fit_clusters(
        embeddings,
        episode_id,
        episode_step,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )
    artifacts.save(output / "clusters.npz")
    (output / "cluster_achievements.json").write_text(
        json.dumps(summarize_cluster_achievements(dataset, artifacts), indent=2),
        encoding="utf-8",
    )
    print(
        f"clusters={len(artifacts.centroids)}, "
        f"threshold={artifacts.threshold:.6f}, edges={int(artifacts.graph.sum())}"
    )

    if args.stop_after_clustering:
        wandb_logger.finish()
        print(f"encoder/clustering pipeline finished in {time.perf_counter() - started:.1f}s")
        return

    graph_kind = "oracle" if oracle_graph is not None else "learned"
    print(f"[5/5] training {graph_kind}-graph-conditioned PPO-RNN")
    goal_runtime = GoalRuntime(
        transition_network=encoder,
        transition_params=encoder_state.params,
        centroids=jnp.asarray(
            artifacts.centroids
            if oracle_graph is None
            else np.zeros((len(oracle_graph.achievement_ids), 1), dtype=np.float32)
        ),
        threshold=artifacts.threshold if oracle_graph is None else 0.0,
        graph=jnp.asarray(
            artifacts.graph if oracle_graph is None else oracle_graph.graph
        ),
        exploration_reward_coefficient=args.exploration_reward_coefficient,
        goal_success_coefficient=args.goal_success_coefficient,
        non_target_achievement_coefficient=(
            args.non_target_achievement_coefficient
        ),
        include_new_tasks=not args.no_new_tasks,
        oracle_achievement_ids=(
            None if oracle_graph is None else oracle_graph.achievement_ids
        ),
    )
    goal_result = jax.jit(
        train_policy(
            goal_config,
            env,
            pixel=args.pixels,
            goal_runtime=goal_runtime,
            dynamic_config=dynamic_config,
            progress_callback=_progress_reporter(
                output, "goal", goal_config, wandb_logger
            ),
            progress_interval=args.log_interval,
        )
    )(jax.random.PRNGKey(args.seed + 3))
    jax.block_until_ready(goal_result["metrics"]["loss"])
    jax.effects_barrier()
    save_params(output / "goal_policy.msgpack", goal_result["runner_state"][0].params)
    if dynamic_config is not None:
        dynamic_state = goal_result["runner_state"][7]
        save_params(
            output / "high_level_q.msgpack",
            {
                "full_q": dynamic_state.q_state.full_q.params,
                "base_q": dynamic_state.q_state.base_q.params,
                "target_full_q": dynamic_state.q_state.target_full_params,
            },
        )
    goal_metrics = _save_metric_summary(
        output / "goal_metrics.json", goal_result["metrics"]
    )
    print(goal_metrics)
    wandb_logger.finish()
    print(f"pipeline finished in {time.perf_counter() - started:.1f}s")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--output", default="runs/sea-craftax-classic")
    observation_group = parser.add_mutually_exclusive_group()
    observation_group.add_argument(
        "--pixels",
        dest="pixels",
        action="store_true",
        help="use raw pixel observations (default and primary experiment)",
    )
    observation_group.add_argument(
        "--symbolic",
        "--symbolic-debug",
        dest="pixels",
        action="store_false",
        help="use symbolic observations and state-conditioned dynamic graph training",
    )
    parser.set_defaults(pixels=True)
    parser.add_argument("--discovery-dataset-checkpoint", help="legacy option; rejected because discovery now requires fresh rollouts")
    parser.add_argument("--base-policy-checkpoint", help="reuse a saved collection policy and skip base PPO training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--wandb-project", help="enable W&B logging for this project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--idle-timeout", type=int, default=100)
    parser.add_argument("--immortal", type=_parse_bool, default=True)
    parser.add_argument("--health-reward-coef", type=float, default=0.0)
    parser.add_argument(
        "--combat-damage-mode",
        choices=("lethal-only", "cumulative"),
        default="lethal-only",
        help="lethal-only preserves SEA combat; cumulative preserves chip damage",
    )
    parser.add_argument("--base-timesteps", type=int, default=200_000_000)
    parser.add_argument("--goal-timesteps", type=int, default=300_000_000)
    parser.add_argument("--discovery-interactions", type=int, default=50_000_000)
    parser.add_argument("--contrast-interactions", type=int, default=1_000_000)
    parser.add_argument("--clustering-transitions", type=int, default=10_000)
    parser.add_argument("--clustering-interactions", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.8)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--reset-ratio", type=int, default=16)
    parser.add_argument("--collect-chunk-steps", type=int, default=64)
    parser.add_argument("--discovery-train-steps", type=int)
    parser.add_argument("--discovery-num-envs", type=int, default=32)
    parser.add_argument("--discovery-batch-size", type=int, default=2560)
    parser.add_argument("--discovery-hidden-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--discovery-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--discovery-achievement-subset",
        choices=("all", "no_eat_plant", "easy17"),
        default="all",
        help=(
            "achievement events used by encoder contrast/prediction and clustering; "
            "excluded achievement-only transitions are masked, not treated as negatives"
        ),
    )
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=20.0,
        help=(
            "summed-loss reference weight; the effective mean-loss coefficient "
            "is contrast_weight * 128 / 2560"
        ),
    )
    parser.add_argument("--contrast-groups", type=int, default=128)
    parser.add_argument("--contrast-episode-capacity", type=int, default=256)
    parser.add_argument("--discovery-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-clusters", type=int)
    parser.add_argument("--max-clusters", type=int, default=30)
    parser.add_argument("--exploration-reward-coefficient", type=float, default=0.1)
    parser.add_argument(
        "--goal-success-coefficient",
        type=float,
        default=5.0,
        help="SEA target-achievement reward scale; default: 5",
    )
    parser.add_argument(
        "--non-target-achievement-coefficient",
        type=float,
        default=1.0,
        help="SEA non-target achievement reward scale; default: 1",
    )
    parser.add_argument("--no-new-tasks", action="store_true")
    parser.add_argument(
        "--oracle-graph",
        "--graph-path",
        dest="oracle_graph",
        help=(
            "achievement-indexed .npz DAG with graph and achievement_ids; "
            "uses the loaded graph in the same goal-training pipeline"
        ),
    )
    parser.add_argument(
        "--high-level-controller",
        choices=("residual", "sea-meta-controller"),
        default="residual",
        help=(
            "symbolic high-level goal selector: configurable Q scoring "
            "(default) or SEA's original graph meta-controller"
        ),
    )
    parser.add_argument(
        "--dynamic-warmup-steps", type=int, default=30_000_000
    )
    parser.add_argument(
        "--dynamic-beta", "--beta", type=float, default=1.0,
        help="beta multiplying the selected Q signal: residual, base-q, or full-q (default: 1)",
    )
    parser.add_argument(
        "--dynamic-lambda", type=float, default=1.0,
        help="non-negative structural prior weight lambda in lambda*A + beta*signal (default: 1)",
    )
    parser.add_argument(
        "--dynamic-score-source", choices=("residual", "base-q", "full-q"), default="residual",
        help="selection signal: residual=Q_F-Q_B (default), base-q=Q_B, full-q=Q_F; Q learning is unchanged",
    )
    parser.add_argument("--dynamic-temperature", type=float, default=1.0)
    parser.add_argument(
        "--allow-unready-goals", type=_parse_bool, default=False, metavar="BOOL",
        help=(
            "true: allow goals with unmet graph prerequisites after warmup; "
            "false: require prerequisites (default). Warmup always excludes unready goals."
        ),
    )
    parser.add_argument(
        "--allow-completed-goals", type=_parse_bool, default=False, metavar="BOOL",
        help=(
            "true: allow goals achieved earlier in the current episode and recognize repeated "
            "achievement events as target successes; false: exclude completed goals (default). "
            "Achievement rewards are first-occurrence-only; repeated success only ends the option. "
            "Requires a symbolic residual controller and an achievement-indexed oracle graph."
        ),
    )
    parser.add_argument(
        "--high-level-replay-capacity", type=int, default=100_000
    )
    parser.add_argument("--high-level-q-batch-size", type=int, default=256)
    parser.add_argument(
        "--high-level-q-learning-rate", type=float, default=1e-4
    )
    parser.add_argument(
        "--high-level-q-replay-ratio", type=float, default=4.0
    )
    parser.add_argument(
        "--high-level-min-replay-size", type=int, default=1024
    )
    parser.add_argument("--high-level-goal-timeout", type=int, default=300)
    parser.add_argument(
        "--stop-after-clustering",
        action="store_true",
        help="save the trained encoder and clustering artifacts without goal-policy training",
    )
    return parser


def main():
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
