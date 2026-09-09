"""Command-line entry point for the Craftax-Classic SEA PPO-RNN pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from craftax.sea.checkpoint import save_params
from craftax.sea.discovery import (
    DiscoveryConfig,
    embed_positive_transitions,
    fit_clusters,
    train_transition_encoder,
)
from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime
from craftax.sea.ppo_rnn import PPOConfig, make_train
from craftax.sea.rollout import collect_transition_dataset


def _last_metrics(metrics):
    return {name: float(np.asarray(value)[-1]) for name, value in metrics.items()}


def run_pipeline(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    symbolic = not args.pixels
    env = make_sea_craftax_classic_env(
        symbolic=symbolic, idle_timeout=args.idle_timeout
    )
    base_config = PPOConfig(
        total_timesteps=args.base_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        hidden_size=args.hidden_size,
        reset_ratio=args.reset_ratio,
    )
    discovery_config = DiscoveryConfig(
        train_steps=args.discovery_train_steps,
        batch_size=args.discovery_batch_size,
        hidden_size=args.discovery_hidden_size,
        embedding_size=args.embedding_size,
    )
    goal_config = PPOConfig(
        total_timesteps=args.goal_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        hidden_size=args.hidden_size,
        reset_ratio=args.reset_ratio,
    )
    (output / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "base_ppo": asdict(base_config),
                "discovery": asdict(discovery_config),
                "goal_ppo": asdict(goal_config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    started = time.perf_counter()
    print("[1/5] training frozen collection policy")
    base_result = jax.jit(make_train(base_config, env, pixel=args.pixels))(
        jax.random.PRNGKey(args.seed)
    )
    jax.block_until_ready(base_result["metrics"]["loss"])
    base_params = base_result["runner_state"][0].params
    save_params(output / "base_policy.msgpack", base_params)
    print(_last_metrics(base_result["metrics"]))

    print("[2/5] collecting frozen-policy transitions")
    dataset = collect_transition_dataset(
        env,
        base_params,
        base_config,
        interactions=args.discovery_interactions,
        pixel=args.pixels,
        seed=args.seed + 1,
        chunk_steps=args.collect_chunk_steps,
        max_random_transitions=args.discovery_dataset_limit,
        max_positive_transitions=args.discovery_positive_limit,
    )
    dataset.save(output / "discovery_dataset.npz")
    positive_count = int((dataset.event_count > 0).sum())
    print(f"collected {len(dataset)} transitions, {positive_count} positive")
    if positive_count < 2:
        raise RuntimeError(
            "fewer than two achievement transitions were collected; "
            "increase --base-timesteps or --discovery-interactions"
        )

    print("[3/5] training SEA transition encoder")
    encoder, encoder_state, encoder_metrics = train_transition_encoder(
        dataset,
        discovery_config,
        pixel=args.pixels,
        seed=args.seed + 2,
    )
    save_params(output / "transition_encoder.msgpack", encoder_state.params)
    print(
        {
            "total": float(encoder_metrics[0]),
            "prediction": float(encoder_metrics[1][0]),
            "contrast": float(encoder_metrics[1][1]),
        }
    )

    print("[4/5] fitting achievement clusters and causal graph")
    embeddings, episode_id, episode_step = embed_positive_transitions(
        encoder, encoder_state.params, dataset
    )
    artifacts = fit_clusters(
        embeddings,
        episode_id,
        episode_step,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )
    artifacts.save(output / "clusters.npz")
    print(
        f"clusters={len(artifacts.centroids)}, "
        f"threshold={artifacts.threshold:.6f}, edges={int(artifacts.graph.sum())}"
    )

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
        make_train(
            goal_config,
            env,
            pixel=args.pixels,
            goal_runtime=goal_runtime,
        )
    )(jax.random.PRNGKey(args.seed + 3))
    jax.block_until_ready(goal_result["metrics"]["loss"])
    save_params(output / "goal_policy.msgpack", goal_result["runner_state"][0].params)
    print(_last_metrics(goal_result["metrics"]))
    print(f"pipeline finished in {time.perf_counter() - started:.1f}s")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idle-timeout", type=int, default=100)
    parser.add_argument("--base-timesteps", type=int, default=200_000_000)
    parser.add_argument("--goal-timesteps", type=int, default=300_000_000)
    parser.add_argument("--discovery-interactions", type=int, default=1_000_000)
    parser.add_argument("--discovery-dataset-limit", type=int, default=200_000)
    parser.add_argument("--discovery-positive-limit", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--reset-ratio", type=int, default=16)
    parser.add_argument("--collect-chunk-steps", type=int, default=64)
    parser.add_argument("--discovery-train-steps", type=int, default=10_000)
    parser.add_argument("--discovery-batch-size", type=int, default=256)
    parser.add_argument("--discovery-hidden-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--min-clusters", type=int)
    parser.add_argument("--max-clusters", type=int, default=30)
    parser.add_argument("--exploration-reward-coefficient", type=float, default=0.1)
    parser.add_argument("--no-new-tasks", action="store_true")
    return parser


def main():
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
