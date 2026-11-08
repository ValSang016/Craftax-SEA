"""Frozen selector evaluation with exactly one independent episode per seed."""
from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, struct

from craftax.sea.dynamic_graph import (
    HighLevelQNetwork, compute_dynamic_edge_scores, get_goal_candidates,
    high_level_option_done, split_symbolic_state, stable_masked_softmax,
)
from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime, GoalState, advance_goal_progress
from craftax.sea.metrics import ACHIEVEMENT_NAMES, HARD_ACHIEVEMENT_INDICES, NO_EAT_PLANT_ACHIEVEMENT_INDICES
from craftax.sea.networks import ActorCriticRNN, ScannedGRU


def score_rates(rates):
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 1 or rates.size == 0 or not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise ValueError("expected a finite vector of achievement success fractions")
    return float(np.expm1(np.log1p(100 * rates).mean()))


def summarize_episodes(achievements, completed):
    achievements = np.asarray(achievements)
    completed = np.asarray(completed, dtype=bool)
    if achievements.ndim != 2 or achievements.shape[1] != 22 or achievements.shape[0] != completed.size:
        raise ValueError("expected all evaluation episodes and all 22 achievement columns")
    if completed.size == 0 or not completed.all():
        raise ValueError("every predetermined episode must finish; do not discard unfinished episodes")
    if not np.isin(achievements, [0, 1]).all():
        raise ValueError("each achievement counts once per episode")
    counts = achievements.astype(np.int64).sum(axis=0)
    rates = counts / completed.size
    hard = np.asarray(HARD_ACHIEVEMENT_INDICES)
    easy = np.asarray([i for i in range(22) if i not in HARD_ACHIEVEMENT_INDICES])
    return dict(num_episodes=int(completed.size), crafter_score22=score_rates(rates),
        crafter_score21_no_eat_plant=score_rates(rates[list(NO_EAT_PLANT_ACHIEVEMENT_INDICES)]),
        achievements_per_episode=float(rates.sum()), easy18_rate=float(rates[easy].mean()),
        hard4_rate=float(rates[hard].mean()),
        achievement_success_count=dict(zip(ACHIEVEMENT_NAMES, counts.tolist())),
        achievement_success_rate=dict(zip(ACHIEVEMENT_NAMES, rates.tolist())))


@struct.dataclass
class EvaluationState:
    env_state: object
    observation: object
    hidden: object
    goals: object
    duration: object
    need_goal: object
    rnn_reset: object
    active: object
    env_keys: object
    goal_keys: object
    action_keys: object
    step: object
    achievements: object
    lengths: object
    terminal_count: object
    terminal_reason: object
    first_achievement_steps: object
    selected_goal_counts: object
    goal_success_count: object
    environment_return: object


def build_evaluator(config, oracle, *, episodes=1024, block=128, protocol="training-env"):
    """Final policy evaluation: no warmup, auto-reset, replay, or optimizer.

    A fixed cohort avoids speed/success-dependent episode sampling. Inactive
    slots perform ignored dummy transitions from their initial valid states.
    """
    args = config["args"]
    dynamic = config["dynamic_graph"]
    if args["pixels"] or not args["no_new_tasks"] or len(oracle.achievement_ids) != 22:
        raise ValueError("this audit evaluates symbolic, 22-goal oracle checkpoints")
    if episodes < 1 or block < 1 or protocol not in ("training-env", "sea-style-termination"):
        raise ValueError("invalid evaluation protocol or cohort size")
    n = episodes
    hidden_size = config["goal_ppo"]["hidden_size"]
    sea_style = protocol == "sea-style-termination"
    env = make_sea_craftax_classic_env(
        symbolic=True, idle_timeout=100 if sea_style else args["idle_timeout"],
        immortal=True if sea_style else args["immortal"],
        health_reward_coef=0.0 if sea_style else args["health_reward_coef"],
        combat_damage_mode=args["combat_damage_mode"],
        track_achievement_events=dynamic.get("allow_completed_goals", False),
    )
    graph = jnp.asarray(oracle.graph)
    runtime = GoalRuntime(transition_network=None, transition_params=None,
        centroids=jnp.zeros((22, 1)), threshold=0., graph=graph,
        include_new_tasks=False, oracle_achievement_ids=oracle.achievement_ids,
        allow_completed_goals=dynamic.get("allow_completed_goals", False))
    policy = ActorCriticRNN(env.num_actions, hidden_size, False, 22)
    full_net = HighLevelQNetwork(22, hidden_size, True)
    base_net = HighLevelQNetwork(22, hidden_size, False)
    reset_many = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_many = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def initialize(seed_start, policy_seed):
        roots = jax.vmap(jax.random.PRNGKey)(jnp.arange(seed_start, seed_start + n, dtype=jnp.uint32))
        fold = jax.vmap(jax.random.fold_in, in_axes=(0, None))
        obs, state = reset_many(fold(roots, 0), env.default_params)
        goals = GoalState(jnp.zeros(n, jnp.int32), jnp.zeros((n, 22), bool), jnp.full(n, 22, jnp.int32))
        carry = EvaluationState(state, obs, ScannedGRU.initialize_carry(n, hidden_size), goals,
            jnp.zeros(n, jnp.int32), jnp.ones(n, bool), jnp.zeros(n, bool), jnp.ones(n, bool),
            fold(roots, 1), fold(fold(roots, 2), policy_seed), fold(fold(roots, 3), policy_seed),
            jnp.int32(0), jnp.zeros((n, 22), bool), jnp.zeros(n, jnp.int32),
            jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32), jnp.full((n, 22), -1, jnp.int32),
            jnp.zeros((n, 22), jnp.int32), jnp.zeros(n, jnp.int32), jnp.zeros(n))
        digest = hashlib.sha256(np.asarray(jax.device_get(obs)).tobytes()).hexdigest()
        return carry, digest

    @jax.jit
    def rollout(carry, policy_params, full_params, base_params, beta, prior_weight, score_source_code):
        initial_env, initial_obs = carry.env_state, carry.observation
        # Captured values may be terminal states in later blocks; retain inactive
        # slots at their already-valid reset states throughout the entire run.
        def step(s, _):
            fold = jax.vmap(jax.random.fold_in, in_axes=(0, None))
            decision = s.active & s.need_goal

            def choose(_):
                external, internal = split_symbolic_state(s.observation)
                full = full_net.apply(full_params, external, s.goals.completed, s.goals.last_completed, internal)
                base = base_net.apply(base_params, external, s.goals.completed, s.goals.last_completed)
                res_scores = compute_dynamic_edge_scores(graph, s.goals.last_completed, full, base, beta,
                    structural_prior_weight=prior_weight, score_source="residual")
                base_scores = compute_dynamic_edge_scores(graph, s.goals.last_completed, full, base, beta,
                    structural_prior_weight=prior_weight, score_source="base-q")
                full_scores = compute_dynamic_edge_scores(graph, s.goals.last_completed, full, base, beta,
                    structural_prior_weight=prior_weight, score_source="full-q")
                candidates = get_goal_candidates(s.goals.completed, graph, False,
                    allow_unready_goals=dynamic.get("allow_unready_goals", False),
                    allow_completed_goals=dynamic.get("allow_completed_goals", False))
                scores = jnp.where(score_source_code == 2, full_scores,
                                   jnp.where(score_source_code == 1, base_scores, res_scores))
                probabilities = stable_masked_softmax(scores, candidates, dynamic["temperature"])
                return jax.vmap(jax.random.categorical)(fold(s.goal_keys, s.step), jnp.log(probabilities))

            proposed = jax.lax.cond(jnp.any(decision), choose, lambda _: s.goals.objective, None)
            objective = jnp.where(decision, proposed, s.goals.objective).astype(jnp.int32)
            goals = s.goals.replace(objective=objective)
            hidden, logits, _ = policy.apply(policy_params, s.hidden, s.observation[None],
                s.rnn_reset[None], objective[None], goals.completed[None])
            action = jax.vmap(jax.random.categorical)(fold(s.action_keys, s.step), logits[0])
            obs, state, reward, done, info = step_many(fold(s.env_keys, s.step), s.env_state, action, env.default_params)
            progress, _, goal_done, *_ = advance_goal_progress(runtime, goals, s.observation, action,
                obs, done, info["event_count"], new_achievements=info["new_achievements"],
                health_reward=reward-info["event_count"], achievement_events=info["achievement_events"])
            option_done = high_level_option_done(goal_done, s.duration + 1, done, dynamic["goal_timeout"])
            ended = s.active & done
            active = s.active & ~done
            def retain(new, reset):
                mask = active.reshape((n,) + (1,) * (new.ndim - 1))
                return jnp.where(mask, new, reset)
            achievements = jnp.where(ended[:, None], info["achievements"], s.achievements)
            lengths = jnp.where(ended, info["episode_length"], s.lengths)
            reason = jnp.where(info["health_done"], 1, jnp.where(info["idle_done"], 2, 3))
            first = jnp.where(s.active[:, None] & info["new_achievements"] & (s.first_achievement_steps < 0),
                s.step + 1, s.first_achievement_steps)
            selections = s.selected_goal_counts + (jax.nn.one_hot(objective, 22, dtype=jnp.int32) * decision[:, None])
            next_state = s.replace(env_state=jax.tree.map(retain, state, initial_env),
                observation=retain(obs, initial_obs), hidden=hidden, goals=progress,
                duration=jnp.where(option_done, 0, s.duration + 1),
                need_goal=active & option_done, rnn_reset=option_done if runtime.reset_rnn_on_goal else done,
                active=active, step=s.step + 1, achievements=achievements, lengths=lengths,
                terminal_count=s.terminal_count + ended.astype(jnp.int32),
                terminal_reason=jnp.where(ended, reason, s.terminal_reason), first_achievement_steps=first,
                selected_goal_counts=selections, goal_success_count=s.goal_success_count + (s.active & goal_done),
                environment_return=s.environment_return + jnp.where(s.active, reward, 0.))
            return next_state, None
        return jax.lax.scan(step, carry, None, length=block)[0]

    protocol_info = dict(protocol=protocol, episodes=episodes, block=block,
        idle_timeout=env.idle_timeout, immortal=env.immortal, health_reward_coef=env.health_reward_coef,
        combat_damage_mode=env.combat_damage_mode, max_timesteps=env.default_params.max_timesteps,
        always_diamond=env.default_params.always_diamond, stochastic_actions=True,
        stochastic_goal_selection=True, warmup_during_evaluation=False,
        optimistic_resets=False, fixed_cohort=True, all_achievements_in_denominator=True)
    return initialize, rollout, protocol_info


def restore(path):
    return jax.tree.map(jnp.asarray, serialization.msgpack_restore(Path(path).read_bytes()))
