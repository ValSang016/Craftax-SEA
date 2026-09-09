"""Reward classification and graph-based goal management for SEA."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct


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
    include_new_tasks: bool = True
    reset_rnn_on_goal: bool = True

    @property
    def num_clusters(self):
        return int(self.centroids.shape[0])

    @property
    def num_objectives(self):
        return self.num_clusters + int(self.include_new_tasks)


def classify_embeddings(embeddings, centroids, threshold):
    distances = jnp.square(embeddings[:, None, :] - centroids[None, :, :]).sum(-1)
    nearest = jnp.argmin(distances, axis=-1)
    min_distance = jnp.take_along_axis(distances, nearest[:, None], axis=1)[:, 0]
    unknown = centroids.shape[0]
    return jnp.where(min_distance < threshold, nearest, unknown), min_distance


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
):
    """Classify an event, shape reward and update the meta-controller."""

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
    shaped_reward = goal_done.astype(jnp.float32) + (
        runtime.exploration_reward_coefficient * event_count
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
    "GoalRuntime",
    "GoalState",
    "apply_goal_transition",
    "classify_embeddings",
    "initialize_goal_state",
    "select_goals",
]
