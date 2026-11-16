"""Goal progress, reward shaping, candidates and option termination.

Nodes are the 22 Craftax-Classic achievements of an oracle graph (no unknown
node).  Ported from the ``dynamic-graph3`` branch (``goal.py`` /
``dynamic_graph.py``) without the original-SEA reproduction options.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from craftax.sea.goal import GoalState, select_goals


@dataclass(frozen=True)
class PlannerGoalRuntime:
    graph: jax.Array
    achievement_ids: tuple[int, ...]
    goal_success_coefficient: float = 3.0
    non_target_achievement_coefficient: float = 1.0
    # Reward for achieving the commanded goal again in the same episode.
    repeat_target_coefficient: float = 0.0
    # Completed goals can be commanded again; success then counts repeats.
    allow_completed_goals: bool = True
    reset_rnn_on_goal: bool = True

    def __post_init__(self):
        coefficients = (
            self.goal_success_coefficient,
            self.non_target_achievement_coefficient,
            self.repeat_target_coefficient,
        )
        if not np.all(np.isfinite(coefficients)) or min(coefficients) < 0:
            raise ValueError("goal reward coefficients must be finite and non-negative")
        if len(self.achievement_ids) != int(self.graph.shape[0]):
            raise ValueError("achievement_ids must match the graph size")

    @property
    def num_goals(self):
        return int(self.graph.shape[0])


def shape_goal_reward(runtime, goal_done, event_count, target_is_new):
    """First-time target, repeated target and other first-time achievements."""
    goal_value = goal_done.astype(jnp.float32)
    first_target = goal_value * target_is_new.astype(jnp.float32)
    repeated_target = goal_value - first_target
    non_target = jnp.maximum(event_count.astype(jnp.float32) - first_target, 0.0)
    return (
        runtime.goal_success_coefficient * first_target
        + runtime.repeat_target_coefficient * repeated_target
        + runtime.non_target_achievement_coefficient * non_target
    )


def advance_goal_progress(
    runtime: PlannerGoalRuntime,
    goal_state: GoalState,
    env_done,
    event_count,
    new_achievements,
    achievement_events,
):
    """Update completion and the current node without choosing a new goal.

    Returns ``(state, reward, goal_done, target_is_new, transition_completed,
    transition_current_node)``.  The returned state is already reset for
    finished episodes; the ``transition_*`` values are the pre-reset ones
    (the outcome of the option that just ended).
    """
    ids = jnp.asarray(runtime.achievement_ids)
    observed = achievement_events if runtime.allow_completed_goals else new_achievements
    events = observed[:, ids]
    novel = new_achievements[:, ids]
    objective = goal_state.objective[:, None]
    goal_done = jnp.take_along_axis(events, objective, axis=1)[:, 0]
    target_is_new = jnp.take_along_axis(novel, objective, axis=1)[:, 0]
    reward = shape_goal_reward(runtime, goal_done, event_count, target_is_new)

    # The current node advances only on a first-time achievement or a target
    # success, never on an incidental repeat (e.g. collecting wood again).
    node_event = novel.any(-1) | goal_done
    node_class = jnp.where(novel.any(-1), jnp.argmax(novel, axis=-1), runtime.num_goals)
    node_class = jnp.where(goal_done, goal_state.objective, node_class)

    completed = goal_state.completed | events
    last_completed = jnp.where(node_event, node_class, goal_state.last_completed)
    transition_completed = completed
    transition_current_node = last_completed

    completed = jnp.where(env_done[:, None], False, completed)
    last_completed = jnp.where(env_done, runtime.num_goals, last_completed)
    state = GoalState(goal_state.objective, completed, last_completed)
    return (
        state,
        reward,
        goal_done,
        target_is_new,
        transition_completed,
        transition_current_node,
    )


def get_goal_candidates(
    completed, graph, *, allow_unready_goals=False, allow_completed_goals=False
):
    """Prerequisite and completion eligibility, with a non-empty fallback."""
    count = graph.shape[0]
    completed = completed[..., :count]
    prerequisites_met = jnp.all(~graph.astype(bool) | completed[..., :, None], axis=-2)
    remaining = jnp.ones_like(completed) if allow_completed_goals else ~completed
    ready = (prerequisites_met | allow_unready_goals) & remaining
    has_ready = jnp.any(ready, axis=-1, keepdims=True)
    has_remaining = jnp.any(remaining, axis=-1, keepdims=True)
    fallback = jnp.where(has_remaining, remaining, jnp.ones_like(remaining))
    return jnp.where(has_ready, ready, fallback)


def high_level_option_done(goal_done, duration, env_done, goal_timeout):
    """After warmup an option ends on target success, timeout or episode end."""
    return goal_done | (duration >= goal_timeout) | env_done


def warmup_option_done(goal_done, event_count, env_done):
    """The existing SEA meta-controller redraws on success, any first-time
    achievement, or episode end (``craftax.sea.goal.apply_goal_transition``)."""
    return goal_done | (event_count > 0) | env_done


def warmup_option_recorded(option_done, goal_done, duration, env_done, goal_timeout):
    """Warmup options replaced for another reason (a non-target first-time
    achievement before the timeout) have an unknown outcome: censored."""
    return option_done & (goal_done | env_done | (duration >= goal_timeout))


def choose_meta_controller_goals(key, completed, current_node, graph):
    """The codebase's existing SEA meta-controller, used during warmup.

    Settings of the existing oracle runs without the unknown goal: no
    reselection of completed goals and no new-task objective.
    """
    keys = jax.random.split(key, completed.shape[0])
    return select_goals(
        keys, completed, current_node, graph,
        include_new_tasks=False, allow_reselect=False, unknown_goal_prob=0.0,
    )


__all__ = [
    "PlannerGoalRuntime",
    "advance_goal_progress",
    "choose_meta_controller_goals",
    "get_goal_candidates",
    "high_level_option_done",
    "shape_goal_reward",
    "warmup_option_done",
    "warmup_option_recorded",
]
