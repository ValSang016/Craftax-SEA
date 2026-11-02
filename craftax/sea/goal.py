"""Reward classification and graph-based goal management for SEA."""

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
    centroids: jax.Array
    threshold: float
    graph: jax.Array
    exploration_reward_coefficient: float = 0.1
    goal_success_coefficient: float = 5.0
    non_target_achievement_coefficient: float | None = 1.0
    include_new_tasks: bool = True
    reset_rnn_on_goal: bool = True
    oracle_achievement_ids: tuple[int, ...] | None = None
    def __post_init__(self):
        coefficients = [self.goal_success_coefficient]
        if self.non_target_achievement_coefficient is not None:
            coefficients.append(self.non_target_achievement_coefficient)
        if not np.all(np.isfinite(coefficients)) or any(
            coefficient < 0 for coefficient in coefficients
        ):
            raise ValueError("goal reward coefficients must be finite and non-negative")

    @property
    def num_clusters(self):
        return int(self.centroids.shape[0])

    @property
    def num_objectives(self):
        return self.num_clusters + int(self.include_new_tasks)


@dataclass(frozen=True)
class OracleGraph:
    """Achievement-indexed DAG loaded from a portable ``.npz`` artifact."""

    graph: np.ndarray
    achievement_ids: tuple[int, ...]
    achievement_names: tuple[str, ...]


def load_oracle_graph(path) -> OracleGraph:
    """Load and validate an achievement-indexed oracle DAG."""

    path = Path(path).expanduser().resolve()
    if path.suffix != ".npz":
        raise ValueError(f"oracle graph must be an .npz file: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        missing = {"graph", "achievement_ids"} - set(data.files)
        if missing:
            raise ValueError(
                f"oracle graph is missing required arrays: {sorted(missing)}"
            )
        graph = np.asarray(data["graph"])
        achievement_ids = np.asarray(data["achievement_ids"])
        names = (
            tuple(str(name) for name in data["achievement_names"].tolist())
            if "achievement_names" in data.files
            else tuple(
                Achievement(int(index)).name.lower() for index in achievement_ids
            )
        )

    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("oracle graph must be a square adjacency matrix")
    node_count = graph.shape[0]
    if achievement_ids.shape != (node_count,):
        raise ValueError("achievement_ids must contain one ID per graph node")
    if len(names) != node_count:
        raise ValueError("achievement_names must contain one name per graph node")
    if not np.issubdtype(achievement_ids.dtype, np.integer):
        raise ValueError("achievement_ids must be integers")
    achievement_ids = achievement_ids.astype(np.int32)
    if len(np.unique(achievement_ids)) != node_count:
        raise ValueError("achievement_ids must be unique")
    if np.any(achievement_ids < 0) or np.any(achievement_ids >= len(Achievement)):
        raise ValueError("achievement ID outside Craftax-Classic range")
    if not np.all((graph == 0) | (graph == 1)):
        raise ValueError("oracle graph entries must be binary")
    graph = graph.astype(np.int8)
    if np.any(np.diag(graph)):
        raise ValueError("oracle graph cannot contain self-edges")

    indegree = graph.sum(axis=0).astype(np.int32)
    ready = list(np.flatnonzero(indegree == 0))
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    if visited != node_count:
        raise ValueError("oracle graph must be a DAG")

    return OracleGraph(
        graph=graph,
        achievement_ids=tuple(int(index) for index in achievement_ids),
        achievement_names=names,
    )


def classify_embeddings(embeddings, centroids, threshold):
    distances = jnp.square(embeddings[:, None, :] - centroids[None, :, :]).sum(-1)
    nearest = jnp.argmin(distances, axis=-1)
    min_distance = jnp.take_along_axis(distances, nearest[:, None], axis=1)[:, 0]
    unknown = centroids.shape[0]
    return jnp.where(min_distance < threshold, nearest, unknown), min_distance


def shape_goal_reward(
    runtime, goal_done, achievement_reward, health_reward=0.0
):
    """Add target, non-target achievement, and signed health rewards."""

    goal_value = goal_done.astype(jnp.float32)
    if runtime.non_target_achievement_coefficient is None:
        achievement_value = (
            runtime.exploration_reward_coefficient * achievement_reward
        )
    else:
        non_target_count = jnp.maximum(achievement_reward - goal_value, 0.0)
        achievement_value = (
            runtime.non_target_achievement_coefficient * non_target_count
        )
    return (
        runtime.goal_success_coefficient * goal_value
        + achievement_value
        + health_reward
    )


def _select_one_goal(key, completed, last_completed, graph, include_new_tasks):
    """JAX equivalent of SEA's GraphObjectiveWrapper sampling rule."""

    cluster_count = graph.shape[0]
    prerequisites_met = jnp.all(
        ~graph.astype(bool) | completed[:cluster_count, None], axis=0
    )
    remaining = ~completed[:cluster_count]
    ready = prerequisites_met & remaining
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
    unready = remaining & ~ready
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
    choose_unknown = include_new_tasks & (jax.random.uniform(unknown_key) < 0.4)
    return jnp.where(choose_unknown, cluster_count, selected).astype(jnp.int32)


def select_goals(keys, completed, last_completed, graph, include_new_tasks=True):
    return jax.vmap(_select_one_goal, in_axes=(0, 0, 0, None, None))(
        keys, completed, last_completed, graph, include_new_tasks
    )


def initialize_goal_state(key, num_envs, graph, include_new_tasks=True):
    num_objectives = graph.shape[0] + int(include_new_tasks)
    completed = jnp.zeros((num_envs, num_objectives), dtype=bool)
    last_completed = jnp.full((num_envs,), num_objectives, dtype=jnp.int32)
    keys = jax.random.split(key, num_envs)
    objective = select_goals(keys, completed, last_completed, graph, include_new_tasks)
    return GoalState(objective, completed, last_completed)


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
    health_reward=0.0,
):
    """Classify an event, shape reward and update the meta-controller."""

    if runtime.oracle_achievement_ids is not None:
        if new_achievements is None:
            raise ValueError("Oracle runtime requires new_achievements")
        known = new_achievements[:, jnp.asarray(runtime.oracle_achievement_ids)]
        unknown = new_achievements.sum(-1) > known.sum(-1)
        events = (
            jnp.concatenate([known, unknown[:, None]], axis=-1)
            if runtime.include_new_tasks
            else known
        )
        event_happened = new_achievements.any(-1)
        goal_done = jnp.take_along_axis(events, goal_state.objective[:, None], axis=1)[:, 0]
        # For simultaneous events, prefer the achieved objective as last event;
        # otherwise use the first known event, or unknown if no known event.
        predicted_class = jnp.where(known.any(-1), jnp.argmax(known, axis=-1), runtime.num_clusters)
        predicted_class = jnp.where(goal_done, goal_state.objective, predicted_class)
        distance = jnp.zeros_like(event_count, dtype=jnp.float32)
        completed_event = (
            events.at[:, runtime.num_clusters].set(False)
            if runtime.include_new_tasks
            else events
        )
        shaped_reward = shape_goal_reward(
            runtime, goal_done, event_count, health_reward
        )
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
        event_happened = event_count > 0
        predicted_class = jnp.where(event_happened, predicted_class, runtime.num_clusters)
        goal_done = event_happened & (predicted_class == goal_state.objective)
        shaped_reward = shape_goal_reward(
            runtime, goal_done, event_count, health_reward
        )

        completed_event = (
            jax.nn.one_hot(predicted_class, runtime.num_objectives, dtype=jnp.bool_)
            & event_happened[:, None]
        )
    completed = goal_state.completed | completed_event
    last_completed = jnp.where(
        event_happened, predicted_class, goal_state.last_completed
    )

    # A physical reset clears option history; a goal event regenerates the
    # objective while preserving the physical Craftax state.
    completed = jnp.where(env_done[:, None], False, completed)
    last_completed = jnp.where(env_done, runtime.num_objectives, last_completed)
    regenerate = event_happened | env_done
    keys = jax.random.split(key, observation.shape[0])
    selected = select_goals(
        keys,
        completed,
        last_completed,
        runtime.graph,
        runtime.include_new_tasks,
    )
    objective = jnp.where(regenerate, selected, goal_state.objective)
    new_goal_state = GoalState(objective, completed, last_completed)
    return new_goal_state, shaped_reward, goal_done, predicted_class, distance


__all__ = [
    "OracleGraph",
    "GoalRuntime",
    "GoalState",
    "apply_goal_transition",
    "classify_embeddings",
    "initialize_goal_state",
    "load_oracle_graph",
    "select_goals",
    "shape_goal_reward",
]
