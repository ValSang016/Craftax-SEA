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

from craftax.sea.checkpoint import save_params
from craftax.sea.discovery import (
    DiscoveryConfig,
    embed_positive_transitions,
    fit_clusters,
    summarize_cluster_achievements,
)
from craftax.sea.env import (
    COMBAT_MODES,
    default_health_reward_coef,
    default_idle_timeout,
    make_sea_craftax_classic_env,
)
from craftax.sea.goal import GoalRuntime, load_oracle_graph
from craftax.sea.metrics import (
    ACHIEVEMENT_NAMES,
    crafter_score_from_rates,
    summarize_training_metrics,
)
from craftax.sea.ppo_rnn import PPOConfig, make_train
from craftax.sea.rollout import iter_policy_transitions
from craftax.sea.streaming_discovery import collect_clustering_dataset, train_streaming_encoder


def _save_metric_summary(path, metrics):
    np.savez_compressed(
        path.with_name(f"{path.stem}_history.npz"),
        **{name: np.asarray(value) for name, value in metrics.items()},
    )
    summary = summarize_training_metrics(metrics)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class WandbLogger:
    """Log windowed PPO metrics, including the 22-achievement Crafter score.

    PPO updates contain few finished episodes, so achievement counts are summed
    over ``log_interval`` updates before success rates and the Crafter score
    are computed.
    """

    _COUNT_KEYS = (
        "achievement_success_count",
        "episode_count",
        "episode_return_sum",
        "episode_length_sum",
    )
    # Per-update ratios that are meaningless when averaged over updates.
    _SKIP_KEYS = (
        "mean_episode_return",
        "mean_episode_length",
        "easy_success_rate",
        "hard_success_rate",
        "crafter_score",
    )

    def __init__(self, args, config, output):
        import wandb

        self.wandb = wandb
        self.interval = args.log_interval
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or output.name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            dir=str(output),
            config=config,
        )

    def stage_callback(self, stage, ppo_config):
        steps_per_update = ppo_config.num_envs * ppo_config.num_steps
        total_updates = ppo_config.total_timesteps // steps_per_update
        self.wandb.define_metric(f"{stage}/env_steps")
        self.wandb.define_metric(f"{stage}/*", step_metric=f"{stage}/env_steps")
        window = {}

        def callback(update, metrics):
            update = int(update)
            for name, value in metrics.items():
                if name in self._SKIP_KEYS:
                    continue
                value = np.asarray(value, dtype=np.float64)
                window[name] = window.get(name, 0.0) + value
            window["_updates"] = window.get("_updates", 0) + 1
            if update % self.interval and update != total_updates:
                return
            record = {f"{stage}/env_steps": update * steps_per_update,
                      f"{stage}/update": update}
            updates = window.pop("_updates")
            counts = {key: window.pop(key) for key in self._COUNT_KEYS}
            for name, value in window.items():
                if value.ndim == 0:
                    record[f"{stage}/{name}"] = float(value / updates)
            window.clear()
            episodes = float(counts["episode_count"])
            record[f"{stage}/episodes_in_window"] = episodes
            if episodes > 0:
                rates = counts["achievement_success_count"] / episodes
                record[f"{stage}/crafter_score"] = crafter_score_from_rates(rates)
                record[f"{stage}/mean_episode_return"] = float(counts["episode_return_sum"] / episodes)
                record[f"{stage}/mean_episode_length"] = float(counts["episode_length_sum"] / episodes)
                for name, rate in zip(ACHIEVEMENT_NAMES, rates):
                    record[f"{stage}/achievements/{name}"] = 100.0 * float(rate)
            self.wandb.log(record)

        return callback

    def log_summary(self, stage, summary):
        for name, value in summary.items():
            if isinstance(value, dict):
                for key, item in value.items():
                    self.run.summary[f"{stage}/final/{name}/{key}"] = item
            else:
                self.run.summary[f"{stage}/final/{name}"] = value

    def finish(self):
        self.wandb.finish()


def _progress_reporter(output, stage, config):
    started = time.perf_counter()
    previous = [None, None]

    def report(update, metrics, params):
        update = int(update)
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
        checkpoint = output / f"{stage}_policy_latest.msgpack"
        temporary = checkpoint.with_suffix(".tmp")
        save_params(temporary, params)
        temporary.replace(checkpoint)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        status = output / "progress.tmp"
        status.write_text(json.dumps(record, indent=2), encoding="utf-8")
        status.replace(output / "progress.json")
        print(
            f"[{stage}] update={update}/{record['total_updates']} steps={steps} "
            f"loss={float(metrics['loss']):.6f} "
            f"return={float(metrics['mean_episode_return']):.3f} "
            f"sps={record.get('steps_per_second', 0):.1f}",
            flush=True,
        )

    return report


def run_pipeline(args):
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
    if args.idle_timeout is None:
        args.idle_timeout = default_idle_timeout(args.mortal)
    if args.health_reward_coef is None:
        args.health_reward_coef = default_health_reward_coef(args.mortal)
    if args.oracle_graph:
        if args.base_only or args.stop_after_clustering or args.base_policy_checkpoint:
            raise ValueError("--oracle-graph trains only the goal policy; drop "
                             "--base-only/--stop-after-clustering/--base-policy-checkpoint")
        oracle_graph, oracle_ids = load_oracle_graph(args.oracle_graph)
    if not 0.0 <= args.unknown_goal_prob <= 1.0:
        raise ValueError("--unknown-goal-prob must be in [0, 1]")
    if args.unknown_goal_prob == 0.0:
        # A never-selected unknown goal is equivalent to no new-task objective.
        args.no_new_tasks = True
    output = Path(args.output)
    if (output / "config.json").exists():
        raise FileExistsError(f"run already exists: {output}; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    symbolic = not args.pixels
    env = make_sea_craftax_classic_env(
        symbolic=symbolic,
        idle_timeout=args.idle_timeout,
        mortal=args.mortal,
        combat=args.combat,
        health_reward_coef=args.health_reward_coef,
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
        "env_name": env.name,
        "base_ppo": asdict(base_config),
        "discovery": asdict(discovery_config),
        "discovery_rollout": asdict(stream_config),
        "discovery_mean_contrast_coefficient": discovery_config.mean_contrast_coefficient,
        "goal_ppo": asdict(goal_config),
    }
    (output / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8",
    )
    logger = WandbLogger(args, run_config, output) if args.wandb else None
    try:
        _run_stages(args, env, output, logger, base_config, stream_config,
                    discovery_config, goal_config,
                    (oracle_graph, oracle_ids) if args.oracle_graph else None)
    finally:
        if logger is not None:
            logger.finish()


def _train_goal_policy(args, env, output, logger, goal_config, goal_runtime, started):
    goal_result = jax.jit(
        make_train(
            goal_config,
            env,
            pixel=args.pixels,
            goal_runtime=goal_runtime,
            progress_callback=_progress_reporter(output, "goal", goal_config),
            progress_interval=args.log_interval,
            metrics_callback=(
                None if logger is None else logger.stage_callback("goal", goal_config)
            ),
        )
    )(jax.random.PRNGKey(args.seed + 3))
    jax.block_until_ready(goal_result["metrics"]["loss"])
    jax.effects_barrier()
    save_params(output / "goal_policy.msgpack", goal_result["runner_state"][0].params)
    goal_metrics = _save_metric_summary(
        output / "goal_metrics.json", goal_result["metrics"]
    )
    if logger is not None:
        logger.log_summary("goal", goal_metrics)
    print(goal_metrics)
    print(f"pipeline finished in {time.perf_counter() - started:.1f}s")


def _run_stages(args, env, output, logger, base_config, stream_config,
                discovery_config, goal_config, oracle):
    train_policy = make_train
    collect_stream = iter_policy_transitions
    started = time.perf_counter()
    goal_options = dict(
        target_reward=args.target_goal_reward,
        nontarget_reward=args.nontarget_goal_reward,
        include_new_tasks=not args.no_new_tasks,
        allow_reselect=args.allow_goal_reselect,
        unknown_goal_prob=args.unknown_goal_prob,
    )
    if oracle is not None:
        graph, achievement_ids = oracle
        print(f"[oracle] training goal PPO-RNN on {args.oracle_graph} "
              f"({len(achievement_ids)} nodes, {int(graph.sum())} edges); "
              "base policy and discovery are skipped")
        goal_runtime = GoalRuntime(
            transition_network=None,
            transition_params=None,
            centroids=None,
            threshold=None,
            graph=jnp.asarray(graph),
            oracle_achievement_ids=achievement_ids,
            **goal_options,
        )
        _train_goal_policy(args, env, output, logger, goal_config, goal_runtime, started)
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
            progress_callback=_progress_reporter(output, "base", base_config),
            progress_interval=args.log_interval,
            metrics_callback=(
                None if logger is None else logger.stage_callback("base", base_config)
            ),
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
        if logger is not None:
            logger.log_summary("base", base_metrics)
        print(base_metrics)

    if args.base_only:
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
        print(f"[encoder] {record}", flush=True)

    print("[2-3/5] collecting fresh transitions and training SEA encoder")
    stream = collect_stream(
        env, base_params, stream_config, interactions=args.discovery_interactions,
        pixel=args.pixels, seed=args.seed + 1, chunk_steps=stream_config.num_steps,
    )
    result = train_streaming_encoder(
        stream, discovery_config, contrast_interactions=args.contrast_interactions,
        pixel=args.pixels, seed=args.seed + 2, progress_callback=encoder_progress,
    )
    if result.summary["prediction_transitions_seen"] != args.discovery_interactions:
        raise RuntimeError("encoder did not consume the full fresh-transition budget")
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
    dataset, clustering_consumed = collect_clustering_dataset(
        clustering_stream, max_transitions=args.clustering_transitions,
    )
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
        print(f"encoder/clustering pipeline finished in {time.perf_counter() - started:.1f}s")
        return

    print("[5/5] training graph-conditioned PPO-RNN")
    goal_runtime = GoalRuntime(
        transition_network=encoder,
        transition_params=encoder_state.params,
        centroids=jnp.asarray(artifacts.centroids),
        threshold=artifacts.threshold,
        graph=jnp.asarray(artifacts.graph),
        **goal_options,
    )
    _train_goal_policy(args, env, output, logger, goal_config, goal_runtime, started)


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
        "--symbolic-debug",
        dest="pixels",
        action="store_false",
        help="use symbolic observations for fast debugging only",
    )
    parser.set_defaults(pixels=True)
    parser.add_argument("--discovery-dataset-checkpoint", help="legacy option; rejected because discovery now requires fresh rollouts")
    parser.add_argument("--base-policy-checkpoint", help="reuse a saved collection policy and skip base PPO training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=None,
        help="steps without a new achievement before the episode ends "
             "(default: 100, or 10000 with --mortal)",
    )
    parser.add_argument(
        "--mortal",
        action="store_true",
        help="use vanilla Craftax health: the player can die (health <= 0 or lava ends the episode)",
    )
    parser.add_argument("--health-reward-coef", type=float, default=None,
                        help="worker reward per health point gained/lost (default: 0.1 if mortal, else 0)")
    parser.add_argument(
        "--combat",
        choices=COMBAT_MODES,
        default="sea",
        help="sea: a hit only damages a creature if it is lethal (cow/skeleton/zombie hp 2/3/5); "
             "org: vanilla Craftax accumulated damage and creature hp (3/3/5)",
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
    parser.add_argument("--target-goal-reward", type=float, default=3.0,
                        help="goal-PPO reward for achieving the current target goal")
    parser.add_argument("--nontarget-goal-reward", type=float, default=1.0,
                        help="goal-PPO reward per first-time unlock of a non-target achievement")
    parser.add_argument("--allow-goal-reselect", action="store_true",
                        help="let the goal selector pick goals that were already achieved "
                             "in the episode; completion then counts repeated events")
    parser.add_argument("--no-new-tasks", action="store_true")
    parser.add_argument(
        "--unknown-goal-prob", type=float, default=0.4,
        help="probability of selecting the unknown/new-task goal (SEA default 0.4; "
             "0 disables it). With --oracle-graph it is achieved by any first-time unlock",
    )
    parser.add_argument(
        "--oracle-graph",
        help="oracle achievement graph (.npz with graph/achievement_ids, e.g. "
             "craftax_classic_22.npz); goals are true achievements, base policy "
             "and discovery are skipped",
    )
    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="craftax-sea")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name", help="run name (default: output directory name)")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
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
