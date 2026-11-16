"""Reward classification and graph-based goal management for SEA.

Goal-PPO reward: ``target_reward`` (default 3) when the current target goal
is achieved, plus ``nontarget_reward`` (default 1) for every other achievement
unlocked for the first time in the episode.

With ``allow_reselect=True`` a goal that was already achieved can be selected
again; goal completion is then detected from every achievement event of a
step (``info["achievement_events"]``), not only from first unlocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from craftax.craftax_classic.constants import Achievement


@struct.dataclass
class GoalState:
    objective: jax.Array
    completed: jax.Array
    last_completed: jax.Array


@dataclass(frozen=True)
class GoalRuntime:
    """Frozen discovery artifacts used by goal-conditioned PPO."""

    transition_network: object
    transition_params: object
    centroids: jax.Array | None
    threshold: float | None
    graph: jax.Array
    target_reward: float = 3.0
    nontarget_reward: float = 1.0
    include_new_tasks: bool = True
    reset_rnn_on_goal: bool = True
    oracle_achievement_ids: tuple[int, ...] | None = None
    allow_reselect: bool = False
    unknown_goal_prob: float = 0.4

    def __post_init__(self):
        if not 0.0 <= self.unknown_goal_prob <= 1.0:
            raise ValueError("unknown_goal_prob must be in [0, 1]")
        if self.oracle_achievement_ids is None:
            if self.transition_network is None or self.centroids is None:
                raise ValueError("learned goal runtime needs an encoder and centroids")
            if int(self.centroids.shape[0]) != int(self.graph.shape[0]):
                raise ValueError("graph and centroid counts differ")
        elif len(self.oracle_achievement_ids) != int(self.graph.shape[0]):
            raise ValueError("oracle_achievement_ids must match the graph size")

    @property
    def num_clusters(self):
        return int(self.graph.shape[0])

    @property
    def num_objectives(self):
        return self.num_clusters + int(self.include_new_tasks)


def classify_embeddings(embeddings, centroids, threshold):
    distances = jnp.square(embeddings[:, None, :] - centroids[None, :, :]).sum(-1)
    nearest = jnp.argmin(distances, axis=-1)
    min_distance = jnp.take_along_axis(distances, nearest[:, None], axis=1)[:, 0]
    unknown = centroids.shape[0]
    return jnp.where(min_distance < threshold, nearest, unknown), min_distance


def _select_one_goal(
    key, completed, last_completed, graph, include_new_tasks, allow_reselect=False,
    unknown_goal_prob=0.4,
):
    """JAX equivalent of SEA's GraphObjectiveWrapper sampling rule.

    ``allow_reselect`` keeps already completed goals as candidates; graph
    prerequisites are still evaluated on the completed set.
    """

    cluster_count = graph.shape[0]
    prerequisites_met = jnp.all(
        ~graph.astype(bool) | completed[:cluster_count, None], axis=0
    )
    remaining = ~completed[:cluster_count]
    candidates = jnp.ones_like(remaining) if allow_reselect else remaining
    ready = prerequisites_met & candidates
    has_last = last_completed < cluster_count
    safe_last = jnp.minimum(last_completed, cluster_count - 1)
    follows_last = graph[safe_last].astype(bool) & has_last

    # Two ready nodes are neighbours if they share an unfinished direct child.
    unfinished_children = graph.astype(bool) & remaining[None, :]
    last_children = unfinished_children[safe_last] & has_last
    shared_child = jnp.any(unfinished_children & last_children[None, :], axis=1)

    continuation = ready & follows_last
    neighbour = ready & ~continuation & shared_child
    jump = ready & ~continuation & ~neighbour
    unready = candidates & ~ready
    categories = jnp.stack([continuation, neighbour, jump, unready])
    category_weights = jnp.array([80.0, 10.0, 10.0, 5.0])
    category_weights *= jnp.any(categories, axis=1)

    category_key, choice_key, unknown_key, fallback_key = jax.random.split(key, 4)
    category = jax.random.categorical(category_key, jnp.log(category_weights + 1e-8))
    choice_weights = categories[category].astype(jnp.float32)
    selected = jax.random.categorical(choice_key, jnp.log(choice_weights + 1e-8))
    fallback_count = cluster_count + int(include_new_tasks)
    fallback = jax.random.randint(fallback_key, (), 0, fallback_count)
    selected = jnp.where(category_weights.sum() > 0, selected, fallback)
    choose_unknown = include_new_tasks & (
        jax.random.uniform(unknown_key) < unknown_goal_prob
    )
    return jnp.where(choose_unknown, cluster_count, selected).astype(jnp.int32)


def select_goals(
    keys, completed, last_completed, graph, include_new_tasks=True, allow_reselect=False,
    unknown_goal_prob=0.4,
):
    return jax.vmap(_select_one_goal, in_axes=(0, 0, 0, None, None, None, None))(
        keys, completed, last_completed, graph, include_new_tasks, allow_reselect,
        unknown_goal_prob,
    )


def initialize_goal_state(
    key, num_envs, graph, include_new_tasks=True, allow_reselect=False,
    unknown_goal_prob=0.4,
):
    num_objectives = graph.shape[0] + int(include_new_tasks)
    completed = jnp.zeros((num_envs, num_objectives), dtype=bool)
    last_completed = jnp.full((num_envs,), num_objectives, dtype=jnp.int32)
    keys = jax.random.split(key, num_envs)
    objective = select_goals(
        keys, completed, last_completed, graph, include_new_tasks, allow_reselect,
        unknown_goal_prob,
    )
    return GoalState(objective, completed, last_completed)


def load_oracle_graph(path):
    """Load an oracle achievement graph such as ``craftax_classic_22.npz``.

    Returns ``(graph, achievement_ids)``; ``graph[i, j] = 1`` means node ``i``
    precedes node ``j`` and node ``i`` is Craftax achievement
    ``achievement_ids[i]``.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        if "graph" not in data.files:
            raise ValueError(f"{path} has no 'graph' array")
        graph = np.asarray(data["graph"]).astype(np.int8)
        if "achievement_ids" in data.files:
            ids = np.asarray(data["achievement_ids"]).astype(np.int64)
        else:
            ids = np.arange(graph.shape[0])
        names = (
            [str(name) for name in data["achievement_names"]]
            if "achievement_names" in data.files
            else None
        )
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("oracle graph must be square")
    if ids.shape != (graph.shape[0],) or len(set(ids.tolist())) != len(ids):
        raise ValueError("achievement_ids must be unique and match the graph size")
    if ids.min() < 0 or ids.max() >= len(Achievement):
        raise ValueError("achievement id outside the Craftax-Classic range")
    if names is not None:
        expected = [Achievement(int(i)).name.lower() for i in ids]
        if names != expected:
            raise ValueError(f"achievement_names do not match ids: {names} != {expected}")
    return graph, tuple(int(i) for i in ids)


def apply_goal_transition(
    runtime: GoalRuntime,
    goal_state: GoalState,
    key,
    observation,
    action,
    next_observation,
    env_done,
    event_count,
    new_achievements=None,
    achievement_events=None,
):
    """Classify an event, shape reward and update the meta-controller.

    ``event_count`` and ``new_achievements`` describe first unlocks within
    the episode.  ``achievement_events`` (needed with ``allow_reselect``)
    also contains repeats of already unlocked achievements.
    """

    if runtime.allow_reselect and achievement_events is None:
        raise ValueError("allow_reselect requires achievement_events")
    event_count = event_count.astype(jnp.float32)
    new_event_happened = event_count > 0
    if runtime.oracle_achievement_ids is not None:
        if new_achievements is None:
            raise ValueError("Oracle runtime requires new_achievements")
        ids = jnp.asarray(runtime.oracle_achievement_ids)
        signal = achievement_events if runtime.allow_reselect else new_achievements

        def per_class(flags):
            known = flags[:, ids]
            unknown = flags.astype(jnp.int32).sum(-1) - known.astype(jnp.int32).sum(-1)
            return known, unknown

        known, _ = per_class(signal)
        # Oracle "unknown" (new-task) goal: any achievement unlocked for the
        # first time in the episode.  This also covers achievements outside a
        # partial oracle graph, and stays achievable when the graph has all 22.
        new_any = new_achievements.any(-1)
        events = jnp.concatenate([known, new_any[:, None]], axis=-1)
        event_happened = signal.any(-1)
        goal_done = jnp.take_along_axis(events, goal_state.objective[:, None], axis=1)[:, 0]
        # For simultaneous events, prefer the achieved known objective as last
        # event; otherwise use the first known event, or unknown if none.
        predicted_class = jnp.where(known.any(-1), jnp.argmax(known, axis=-1), runtime.num_clusters)
        known_goal_done = goal_done & (goal_state.objective < runtime.num_clusters)
        predicted_class = jnp.where(known_goal_done, goal_state.objective, predicted_class)
        distance = jnp.zeros_like(event_count, dtype=jnp.float32)
        # Drop the unknown column; it is only an objective with new tasks.
        completed_event = events.at[:, runtime.num_clusters].set(False)[
            :, : runtime.num_objectives
        ]

        # Exact count of first unlocks that belong to the achieved target.
        new_known, _ = per_class(new_achievements)
        new_per_class = jnp.concatenate(
            [new_known.astype(jnp.float32), new_any[:, None].astype(jnp.float32)],
            axis=-1,
        )
        target_new = jnp.take_along_axis(
            new_per_class, goal_state.objective[:, None], axis=1
        )[:, 0] * goal_done
    else:
        _, embeddings = runtime.transition_network.apply(
            runtime.transition_params,
            observation,
            action,
            next_observation,
            env_done,
        )
        predicted_class, distance = classify_embeddings(
            embeddings, runtime.centroids, runtime.threshold
        )
        event_happened = (
            achievement_events.any(-1) if runtime.allow_reselect else new_event_happened
        )
        predicted_class = jnp.where(event_happened, predicted_class, runtime.num_clusters)
        goal_done = event_happened & (predicted_class == goal_state.objective)
        # The encoder labels one event per transition; when a first unlock
        # happened alongside the target, count one of them as the target.
        target_new = (goal_done & new_event_happened).astype(jnp.float32)

        completed_event = (
            jax.nn.one_hot(predicted_class, runtime.num_objectives, dtype=jnp.bool_)
            & event_happened[:, None]
        )
    nontarget_count = jnp.maximum(event_count - target_new, 0.0)
    shaped_reward = (
        runtime.target_reward * goal_done.astype(jnp.float32)
        + runtime.nontarget_reward * nontarget_count
    )
    completed = goal_state.completed | completed_event
    last_completed = jnp.where(
        event_happened, predicted_class, goal_state.last_completed
    )

    # A physical reset clears option history; a goal event regenerates the
    # objective while preserving the physical Craftax state.  Without
    # reselection every event is a first unlock, so this is SEA's rule.
    completed = jnp.where(env_done[:, None], False, completed)
    last_completed = jnp.where(env_done, runtime.num_objectives, last_completed)
    regenerate = goal_done | new_event_happened | env_done
    keys = jax.random.split(key, observation.shape[0])
    selected = select_goals(
        keys,
        completed,
        last_completed,
        runtime.graph,
        runtime.include_new_tasks,
        runtime.allow_reselect,
        runtime.unknown_goal_prob,
    )
    objective = jnp.where(regenerate, selected, goal_state.objective)
    new_goal_state = GoalState(objective, completed, last_completed)
    return new_goal_state, shaped_reward, goal_done, predicted_class, distance


__all__ = [
    "GoalRuntime",
    "GoalState",
    "apply_goal_transition",
    "classify_embeddings",
    "initialize_goal_state",
    "load_oracle_graph",
    "select_goals",
]
