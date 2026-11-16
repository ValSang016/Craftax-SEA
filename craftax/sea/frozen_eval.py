"""Held-out evaluation of a finished SEA / graph-planner run with frozen weights.

The final ``goal_policy.msgpack`` (and, for the graph planner,
``high_level_q.msgpack``) are loaded and the training rollout code is run with
no PPO or graph-model updates on seeds never used for training.  Goal
selection is the run's own controller with no warmup: the SEA meta-controller
for exp. 1, the K-hop planner for exp. 2.  Episodes that finish within the
step budget are scored; episodes still running at the end are not counted.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from craftax.sea.dynamic_graph import DynamicGraphConfig
from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime, load_oracle_graph
from craftax.sea.metrics import (
    ACHIEVEMENT_NAMES,
    HARD_ACHIEVEMENT_INDICES,
    NO_EAT_PLANT_ACHIEVEMENT_INDICES,
)
from craftax.sea.ppo_rnn import PPOConfig, make_train

DEFAULT_SEEDS = (84001, 84002, 84003, 84004, 84005)


def _score(rates):
    return float(np.expm1(np.log1p(100.0 * np.asarray(rates)).mean()))


def summarize_counts(achievement_counts, episodes):
    rates = np.asarray(achievement_counts, dtype=np.float64) / max(int(episodes), 1)
    return {
        "episodes": int(episodes),
        "crafter_score22": _score(rates),
        "crafter_score21": _score(rates[list(NO_EAT_PLANT_ACHIEVEMENT_INDICES)]),
        "hard4_success_rate": float(rates[list(HARD_ACHIEVEMENT_INDICES)].mean()),
        "achievement_success_rate": dict(zip(ACHIEVEMENT_NAMES, rates.tolist())),
    }


def _rollout_shape(num_envs, steps_per_seed, num_steps):
    """Pick ``updates`` so updates * num_steps * num_envs is closest to the budget."""
    per_env = steps_per_seed / num_envs
    updates = max(1, round(per_env / num_steps))
    return updates, updates * num_steps * num_envs


def build_frozen_evaluator(run_dir, *, num_envs=256, steps_per_seed=1_000_000, num_steps=63):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    args = config["args"]
    if args.get("pixels", True):
        raise ValueError("frozen evaluation supports symbolic runs only")
    oracle = load_oracle_graph(run_dir / "oracle_graph.npz")

    dynamic_dict = config.get("dynamic_graph")
    dynamic_config = None
    if dynamic_dict is not None:
        dynamic_config = DynamicGraphConfig(**dynamic_dict)
        if dynamic_config.controller != "graph-planner":
            raise ValueError("frozen evaluation supports SEA and graph-planner runs")
        # No warmup: the final controller selects every goal.  A minimal replay
        # keeps memory small; it is never sampled because nothing is trained.
        dynamic_config = replace(
            dynamic_config, warmup_steps=0, replay_capacity=dynamic_config.batch_size,
            min_replay_size=dynamic_config.batch_size,
        )

    allow_completed = bool(args.get("allow_completed_goals", False))
    env = make_sea_craftax_classic_env(
        symbolic=True,
        idle_timeout=args["idle_timeout"],
        immortal=args["immortal"],
        health_reward_coef=args.get("health_reward_coef", 0.0),
        combat_damage_mode=args.get("combat_damage_mode", "lethal-only"),
        track_achievement_events=allow_completed,
    )
    runtime = GoalRuntime(
        transition_network=None,
        transition_params=None,
        centroids=jnp.zeros((len(oracle.achievement_ids), 1), dtype=jnp.float32),
        threshold=0.0,
        graph=jnp.asarray(oracle.graph),
        exploration_reward_coefficient=args.get("exploration_reward_coefficient", 0.1),
        goal_success_coefficient=args.get("goal_success_coefficient", 5.0),
        non_target_achievement_coefficient=args.get(
            "non_target_achievement_coefficient", 1.0
        ),
        include_new_tasks=not args.get("no_new_tasks", False),
        oracle_achievement_ids=oracle.achievement_ids,
        allow_completed_goals=allow_completed,
        repeat_target_coefficient=args.get("repeat_target_coefficient", 0.0),
        # Runs created before these options used the earlier port's rules.
        regenerate_on_any_event=args.get("sea_regenerate", "any") == "any",
        per_node_category_weights=args.get("sea_category_weights", "per-category")
        == "per-node",
    )

    updates, total_steps = _rollout_shape(num_envs, steps_per_seed, num_steps)
    ppo = PPOConfig(**config["goal_ppo"])
    ppo = replace(
        ppo, num_envs=num_envs, num_steps=num_steps,
        total_timesteps=updates * num_steps * num_envs,
        num_minibatches=math.gcd(ppo.num_minibatches, num_envs),
        reset_ratio=math.gcd(ppo.reset_ratio, num_envs),
    )

    frozen = {"policy": serialization.msgpack_restore(
        (run_dir / "goal_policy.msgpack").read_bytes()
    )}
    if dynamic_config is not None:
        high_level = serialization.msgpack_restore(
            (run_dir / "high_level_q.msgpack").read_bytes()
        )
        frozen["option_model"] = high_level["option_model"]
        frozen["option_model_target"] = high_level["option_model_target"]
    frozen = jax.tree.map(jnp.asarray, frozen)

    train = make_train(
        ppo, env, pixel=False, goal_runtime=runtime,
        dynamic_config=dynamic_config, frozen_params=frozen,
    )

    @jax.jit
    def evaluate(key):
        metrics = train(key)["metrics"]
        return {
            "episode_count": metrics["episode_count"].sum(),
            "achievement_success_count": metrics["achievement_success_count"].sum(0),
            "episode_length_sum": metrics["episode_length_sum"].sum(),
        }

    return evaluate, {
        "controller": "graph-planner" if dynamic_config else "sea-meta-controller",
        "num_envs": num_envs,
        "steps_per_seed": total_steps,
        "rollout_steps_per_env": updates * num_steps,
    }


def evaluate_run(run_dir, seeds=DEFAULT_SEEDS, **kwargs):
    evaluate, setup = build_frozen_evaluator(run_dir, **kwargs)
    per_seed = []
    pooled_counts = np.zeros(len(ACHIEVEMENT_NAMES))
    pooled_episodes = 0
    for seed in seeds:
        result = jax.device_get(evaluate(jax.random.PRNGKey(seed)))
        episodes = int(result["episode_count"])
        counts = np.asarray(result["achievement_success_count"], dtype=np.float64)
        summary = summarize_counts(counts, episodes)
        summary["seed"] = int(seed)
        summary["mean_episode_length"] = float(result["episode_length_sum"]) / max(episodes, 1)
        per_seed.append(summary)
        pooled_counts += counts
        pooled_episodes += episodes
    scores22 = np.array([s["crafter_score22"] for s in per_seed])
    scores21 = np.array([s["crafter_score21"] for s in per_seed])
    return {
        "run": str(Path(run_dir)),
        "protocol": "frozen final checkpoint, held-out env seeds, no updates",
        **setup,
        "seeds": list(map(int, seeds)),
        "per_seed": per_seed,
        "mean_over_seeds": {
            "crafter_score22": float(scores22.mean()),
            "crafter_score22_std": float(scores22.std(ddof=1)) if len(seeds) > 1 else 0.0,
            "crafter_score21": float(scores21.mean()),
            "crafter_score21_std": float(scores21.std(ddof=1)) if len(seeds) > 1 else 0.0,
        },
        "pooled": summarize_counts(pooled_counts, pooled_episodes),
    }


__all__ = ["DEFAULT_SEEDS", "build_frozen_evaluator", "evaluate_run", "summarize_counts"]
