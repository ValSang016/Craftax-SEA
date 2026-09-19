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
from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime
from craftax.sea.metrics import summarize_training_metrics
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
    train_policy = make_train
    collect_stream = iter_policy_transitions
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
    symbolic = not args.pixels
    env = make_sea_craftax_classic_env(
        symbolic=symbolic, idle_timeout=args.idle_timeout
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
    (output / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "base_ppo": asdict(base_config),
                "discovery": asdict(discovery_config),
                "discovery_rollout": asdict(stream_config),
                "discovery_mean_contrast_coefficient": discovery_config.mean_contrast_coefficient,
                "goal_ppo": asdict(goal_config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    started = time.perf_counter()
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
        exploration_reward_coefficient=args.exploration_reward_coefficient,
        include_new_tasks=not args.no_new_tasks,
    )
    goal_result = jax.jit(
        train_policy(
            goal_config,
            env,
            pixel=args.pixels,
            goal_runtime=goal_runtime,
            progress_callback=_progress_reporter(output, "goal", goal_config),
            progress_interval=args.log_interval,
        )
    )(jax.random.PRNGKey(args.seed + 3))
    jax.block_until_ready(goal_result["metrics"]["loss"])
    jax.effects_barrier()
    save_params(output / "goal_policy.msgpack", goal_result["runner_state"][0].params)
    goal_metrics = _save_metric_summary(
        output / "goal_metrics.json", goal_result["metrics"]
    )
    print(goal_metrics)
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
    parser.add_argument("--idle-timeout", type=int, default=100)
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
    parser.add_argument("--exploration-reward-coefficient", type=float, default=0.1)
    parser.add_argument("--no-new-tasks", action="store_true")
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
