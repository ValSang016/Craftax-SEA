"""Train only the SEA goal policy from frozen discovery artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

from craftax.sea.checkpoint import save_params
from craftax.sea.discovery import ClusterArtifacts
from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime, load_oracle_graph
from craftax.sea.networks import TransitionEncoder
from craftax.sea.ppo_rnn import PPOConfig, make_train
from craftax.sea.train import (
    _WandbLogger,
    _parse_bool,
    _progress_reporter,
    _save_metric_summary,
)


def _saved_or_default(saved_args, name, default):
    value = saved_args.get(name)
    return default if value is None else value


def run(args):
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser()
    inherited_oracle = source / "oracle_graph.npz"
    oracle_source = (
        Path(args.oracle_graph).expanduser().resolve()
        if args.oracle_graph is not None
        else inherited_oracle if inherited_oracle.is_file() else None
    )
    oracle_graph = (
        load_oracle_graph(oracle_source) if oracle_source is not None else None
    )
    required = (
        "config.json",
        "transition_encoder.msgpack",
        "clusters.npz",
    )
    missing = [source / name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("missing source artifacts: " + ", ".join(map(str, missing)))
    if (output / "config.json").exists() or output.exists():
        raise FileExistsError(f"output already exists: {output}")

    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    saved_args = source_config["args"]
    pixel = bool(_saved_or_default(saved_args, "pixels", True))
    seed = int(_saved_or_default(saved_args, "seed", 0) if args.seed is None else args.seed)
    immortal = bool(
        _saved_or_default(saved_args, "immortal", True)
        if args.immortal is None
        else args.immortal
    )
    health_coef = float(
        _saved_or_default(saved_args, "health_reward_coef", 0.0)
        if args.health_reward_coef is None
        else args.health_reward_coef
    )
    idle_timeout = int(
        _saved_or_default(saved_args, "idle_timeout", 100)
        if args.idle_timeout is None
        else args.idle_timeout
    )
    combat_damage_mode = (
        _saved_or_default(saved_args, "combat_damage_mode", "lethal-only")
        if args.combat_damage_mode is None
        else args.combat_damage_mode
    )
    no_new_tasks = (
        bool(_saved_or_default(saved_args, "no_new_tasks", False))
        if args.no_new_tasks is None
        else args.no_new_tasks
    )
    goal_success_coefficient = float(args.goal_success_coefficient)
    non_target_achievement_coefficient = float(
        args.non_target_achievement_coefficient
    )

    goal_config = PPOConfig(**source_config["goal_ppo"])
    if args.goal_timesteps is not None:
        goal_config = replace(goal_config, total_timesteps=args.goal_timesteps)

    discovery = source_config["discovery"]
    encoder = TransitionEncoder(
        action_dim=17,
        hidden_size=int(discovery["hidden_size"]),
        embedding_size=int(discovery["embedding_size"]),
        pixel=pixel,
    )
    encoder_params = serialization.msgpack_restore(
        (source / "transition_encoder.msgpack").read_bytes()
    )
    artifacts = ClusterArtifacts.load(source / "clusters.npz")

    output.mkdir(parents=True)
    for name in (
        "base_policy.msgpack",
        "transition_encoder.msgpack",
        "clusters.npz",
        "cluster_achievements.json",
        "discovery_summary.json",
        "clustering_collection.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    if oracle_source is not None:
        shutil.copy2(oracle_source, output / "oracle_graph.npz")

    run_config = source_config
    run_config["args"].update(
        {
            "output": str(output),
            "seed": seed,
            "immortal": immortal,
            "health_reward_coef": health_coef,
            "idle_timeout": idle_timeout,
            "combat_damage_mode": combat_damage_mode,
            "goal_timesteps": goal_config.total_timesteps,
            "goal_only_from": str(source),
            "oracle_graph": "oracle_graph.npz" if oracle_graph is not None else None,
            "no_new_tasks": no_new_tasks,
            "goal_success_coefficient": goal_success_coefficient,
            "non_target_achievement_coefficient": (
                non_target_achievement_coefficient
            ),
        }
    )
    run_config["goal_ppo"] = vars(goal_config)
    (output / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    args.output = str(output)
    wandb_logger = _WandbLogger(args, run_config)
    env = make_sea_craftax_classic_env(
        symbolic=not pixel,
        idle_timeout=idle_timeout,
        immortal=immortal,
        health_reward_coef=health_coef,
        combat_damage_mode=combat_damage_mode,
    )
    runtime = GoalRuntime(
        transition_network=encoder,
        transition_params=encoder_params,
        centroids=(
            jnp.asarray(artifacts.centroids)
            if oracle_graph is None
            else jnp.zeros((len(oracle_graph.achievement_ids), 1), dtype=jnp.float32)
        ),
        threshold=artifacts.threshold if oracle_graph is None else 0.0,
        graph=jnp.asarray(
            artifacts.graph if oracle_graph is None else oracle_graph.graph
        ),
        exploration_reward_coefficient=float(
            _saved_or_default(saved_args, "exploration_reward_coefficient", 0.1)
        ),
        goal_success_coefficient=goal_success_coefficient,
        non_target_achievement_coefficient=(
            non_target_achievement_coefficient
        ),
        include_new_tasks=not no_new_tasks,
        oracle_achievement_ids=(
            None if oracle_graph is None else oracle_graph.achievement_ids
        ),
    )

    print(
        f"[5/5] goal-only training from {source}; nodes={runtime.num_clusters}, "
        f"seed={seed}, immortal={immortal}, idle_timeout={idle_timeout}, "
        f"combat_damage_mode={combat_damage_mode}, "
        f"graph={'oracle' if oracle_graph is not None else 'learned'}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        result = jax.jit(
            make_train(
                goal_config,
                env,
                pixel=pixel,
                goal_runtime=runtime,
                progress_callback=_progress_reporter(
                    output, "goal", goal_config, wandb_logger
                ),
                progress_interval=args.log_interval,
            )
        )(jax.random.PRNGKey(seed + 3))
        jax.block_until_ready(result["metrics"]["loss"])
        jax.effects_barrier()
        save_params(output / "goal_policy.msgpack", result["runner_state"][0].params)
        metrics = _save_metric_summary(output / "goal_metrics.json", result["metrics"])
        print(metrics)
        print(f"goal-only training finished in {time.perf_counter() - started:.1f}s")
    finally:
        wandb_logger.finish()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="completed SEA run containing encoder and clusters")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--goal-timesteps", type=int)
    parser.add_argument(
        "--goal-success-coefficient",
        type=float,
        default=5.0,
        help="target-achievement reward scale; default: 5",
    )
    parser.add_argument(
        "--non-target-achievement-coefficient",
        type=float,
        default=1.0,
        help="non-target achievement reward scale; default: 1",
    )
    parser.add_argument(
        "--no-new-tasks",
        action="store_true",
        default=None,
        help="disable the separate unknown/new-task objective",
    )
    parser.add_argument(
        "--oracle-graph",
        help=(
            "achievement-indexed .npz DAG with graph and achievement_ids; "
            "defaults to source/oracle_graph.npz when present"
        ),
    )
    parser.add_argument("--idle-timeout", type=int)
    parser.add_argument("--immortal", type=_parse_bool)
    parser.add_argument("--health-reward-coef", type=float)
    parser.add_argument(
        "--combat-damage-mode", choices=("lethal-only", "cumulative")
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
