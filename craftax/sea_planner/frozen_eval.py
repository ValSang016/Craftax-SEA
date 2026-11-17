"""Held-out evaluation of finished runs with frozen weights.

The final checkpoint runs on env seeds never used for training (default
84001-84005, about 1M steps each: 256 envs x 62 updates x 63 steps) with no
PPO or option-model updates.  Episodes finishing inside the budget are scored.

Supported runs:
* graph planner (``craftax.sea_planner.train``): the K-hop planner selects
  every goal (no warmup);
* SEA baseline with an oracle graph (``craftax.sea.train --oracle-graph``):
  the existing SEA meta-controller (``craftax.sea.goal.apply_goal_transition``).
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

from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import (
    GoalRuntime,
    apply_goal_transition,
    initialize_goal_state,
    load_oracle_graph,
)
from craftax.sea.metrics import ACHIEVEMENT_NAMES
from craftax.sea.networks import ActorCriticRNN, ScannedGRU
from craftax.sea.ppo_rnn import PPOConfig
from craftax.sea.vector_env import SeaAutoResetVecEnv
from craftax.sea_planner.metrics import scores_from_counts

DEFAULT_SEEDS = (84001, 84002, 84003, 84004, 84005)


def rollout_shape(num_envs, steps_per_seed, num_steps):
    updates = max(1, round(steps_per_seed / num_envs / num_steps))
    return updates, updates * num_steps * num_envs


def _restore(path):
    return jax.tree.map(jnp.asarray, serialization.msgpack_restore(Path(path).read_bytes()))


def _planner_evaluator(run_dir, config, num_envs, updates, num_steps):
    from craftax.sea_planner.config import PlannerConfig
    from craftax.sea_planner.ppo import make_train
    from craftax.sea_planner.train import make_env, make_runtime

    args = config["args"]
    planner = PlannerConfig(**config["planner"])
    # No warmup: the planner selects every goal.  The replay is never sampled.
    planner = replace(
        planner, warmup_steps=0, planner_epsilon=0.0, replay_capacity=planner.batch_size,
        min_replay_size=planner.batch_size,
    )
    ppo = PPOConfig(**config["goal_ppo"])
    ppo = replace(
        ppo, num_envs=num_envs, num_steps=num_steps,
        total_timesteps=updates * num_steps * num_envs,
        num_minibatches=math.gcd(ppo.num_minibatches, num_envs),
        reset_ratio=math.gcd(ppo.reset_ratio, num_envs),
    )
    high_level = _restore(run_dir / "high_level_q.msgpack")
    frozen = {
        "policy": _restore(run_dir / "goal_policy.msgpack"),
        "option_model": high_level["option_model"],
        "option_model_target": high_level["option_model_target"],
    }
    train = make_train(
        ppo, make_env(args), make_runtime(args, run_dir / "oracle_graph.npz"), planner,
        frozen_params=frozen,
    )

    def evaluate(key):
        metrics = train(key)["metrics"]
        return {
            "episode_count": metrics["episode_count"].sum(),
            "achievement_success_count": metrics["achievement_success_count"].sum(0),
            "episode_length_sum": metrics["episode_length_sum"].sum(),
        }

    return evaluate, "graph-planner"


def _sea_evaluator(run_dir, config, num_envs, updates, num_steps):
    args = config["args"]
    if args.get("pixels", True) is not False or not args.get("oracle_graph"):
        raise ValueError("SEA baseline evaluation needs a symbolic --oracle-graph run")
    env = make_sea_craftax_classic_env(
        symbolic=True, idle_timeout=args["idle_timeout"],
        mortal=args.get("mortal", False), combat=args.get("combat", "sea"),
    )
    graph_path = Path(args["oracle_graph"])
    if not graph_path.is_absolute():
        graph_path = Path(__file__).resolve().parents[2] / graph_path
    graph, ids = load_oracle_graph(graph_path)
    runtime = GoalRuntime(
        transition_network=None, transition_params=None, centroids=None, threshold=None,
        graph=jnp.asarray(graph),
        target_reward=args.get("target_goal_reward", 3.0),
        nontarget_reward=args.get("nontarget_goal_reward", 1.0),
        include_new_tasks=not args.get("no_new_tasks", False),
        oracle_achievement_ids=ids,
        allow_reselect=args.get("allow_goal_reselect", False),
        unknown_goal_prob=args.get("unknown_goal_prob", 0.4),
    )
    params = env.default_params
    vec_env = SeaAutoResetVecEnv(env, num_envs, math.gcd(args.get("reset_ratio", 16), num_envs))
    network = ActorCriticRNN(
        action_dim=env.action_space(params).n,
        hidden_size=config["goal_ppo"]["hidden_size"],
        pixel=False,
        num_objectives=runtime.num_objectives,
    )
    policy = _restore(run_dir / "goal_policy.msgpack")
    total_steps = updates * num_steps

    def evaluate(key):
        key, reset_key, goal_key = jax.random.split(key, 3)
        observation, env_state = vec_env.reset(reset_key, params)
        goal_state = initialize_goal_state(
            goal_key, num_envs, runtime.graph, runtime.include_new_tasks,
            runtime.allow_reselect, runtime.unknown_goal_prob,
        )
        hidden = ScannedGRU.initialize_carry(num_envs, config["goal_ppo"]["hidden_size"])
        zeros = jnp.zeros((len(ACHIEVEMENT_NAMES),), dtype=jnp.int32)
        carry = (env_state, observation, jnp.zeros((num_envs,), bool), hidden,
                 goal_state, key, zeros, jnp.int32(0), jnp.float32(0.0))

        def step(carry, _):
            (env_state, observation, reset, hidden, goal_state, key,
             counts, episodes, lengths) = carry
            key, action_key, step_key, goal_key = jax.random.split(key, 4)
            hidden, logits, _ = network.apply(
                policy, hidden, observation[None], reset[None],
                goal_state.objective[None], goal_state.completed[None],
            )
            action = jax.random.categorical(action_key, logits[0])
            next_observation, env_state, reward, done, info = vec_env.step(
                step_key, env_state, action, params
            )
            goal_state, _, goal_done, _, _ = apply_goal_transition(
                runtime, goal_state, goal_key, observation, action, info["terminal_obs"],
                done, reward, new_achievements=info["new_achievements"],
                achievement_events=info["achievement_events"],
            )
            counts = counts + (info["achievements"] & done[:, None]).sum(0)
            episodes = episodes + done.sum()
            lengths = lengths + (info["episode_length"] * done).sum()
            return (env_state, next_observation, done | goal_done, hidden, goal_state,
                    key, counts, episodes, lengths), None

        carry, _ = jax.lax.scan(step, carry, None, length=total_steps)
        return {
            "episode_count": carry[7],
            "achievement_success_count": carry[6],
            "episode_length_sum": carry[8],
        }

    return evaluate, "sea-meta-controller"


def build_frozen_evaluator(run_dir, *, num_envs=256, steps_per_seed=1_000_000, num_steps=63):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    updates, total = rollout_shape(num_envs, steps_per_seed, num_steps)
    builder = _planner_evaluator if config.get("method") == "graph-planner" else _sea_evaluator
    evaluate, controller = builder(run_dir, config, num_envs, updates, num_steps)
    return jax.jit(evaluate), {
        "controller": controller,
        "num_envs": num_envs,
        "steps_per_seed": total,
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
        summary = scores_from_counts(counts, episodes)
        summary["seed"] = int(seed)
        summary["mean_episode_length"] = float(result["episode_length_sum"]) / max(episodes, 1)
        per_seed.append(summary)
        pooled_counts += counts
        pooled_episodes += episodes
    scores22 = np.array([s["crafter_score22"] for s in per_seed])
    scores21 = np.array([s["crafter_score21"] for s in per_seed])
    std = (lambda x: float(x.std(ddof=1))) if len(seeds) > 1 else (lambda x: 0.0)
    return {
        "run": str(Path(run_dir)),
        "protocol": "frozen final checkpoint, held-out env seeds, no updates",
        **setup,
        "seeds": [int(seed) for seed in seeds],
        "per_seed": per_seed,
        "mean_over_seeds": {
            "crafter_score22": float(scores22.mean()),
            "crafter_score22_std": std(scores22),
            "crafter_score21": float(scores21.mean()),
            "crafter_score21_std": std(scores21),
        },
        "pooled": scores_from_counts(pooled_counts, pooled_episodes),
    }


__all__ = ["DEFAULT_SEEDS", "build_frozen_evaluator", "evaluate_run", "rollout_shape"]
