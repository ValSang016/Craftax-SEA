"""State-conditioned high-level Q learning for symbolic SEA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

# Craftax-classic # TODO : full craftax / pixel에서도 적용할 수 있도록
SYMBOLIC_OBSERVATION_SIZE = 1345
INTERNAL_STATE_START = 1335
INTERNAL_STATE_STOP = 1339


def split_symbolic_state(state):
    """Split a Classic symbolic observation into external and internal state."""

    if state.shape[-1] != SYMBOLIC_OBSERVATION_SIZE:
        raise ValueError(
            "dynamic graph training requires Craftax-Classic symbolic "
            f"observations of width {SYMBOLIC_OBSERVATION_SIZE}, got {state.shape[-1]}"
        )
    internal_state = state[..., INTERNAL_STATE_START:INTERNAL_STATE_STOP]
    external_state = jnp.concatenate(
        [state[..., :INTERNAL_STATE_START], state[..., INTERNAL_STATE_STOP:]],
        axis=-1,
    )
    return external_state, internal_state


@dataclass(frozen=True)
class DynamicGraphConfig:
    warmup_steps: int = 30_000_000
    residual_weight: float = 1.0
    temperature: float = 1.0
    replay_capacity: int = 100_000
    batch_size: int = 256
    learning_rate: float = 1e-4
    high_level_q_replay_ratio: float = 4.0
    min_replay_size: int = 1024
    goal_timeout: int = 300
    allow_unready_goals: bool = False
    allow_completed_goals: bool = False
    structural_prior_weight: float = 1.0
    score_source: str = "residual"

    def validate(self): 
        # TODO 이거 정리할 수 있으면 하기;
        if self.warmup_steps < 0:
            raise ValueError("dynamic warmup steps cannot be negative")
        if not np.isfinite(self.residual_weight):
            raise ValueError("dynamic residual weight must be finite")
        if not np.isfinite(self.structural_prior_weight) or self.structural_prior_weight < 0:
            raise ValueError("structural prior weight must be finite and non-negative")
        if self.score_source not in ("residual", "base-q", "full-q"):
            raise ValueError("dynamic score source must be residual, base-q, or full-q")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("dynamic softmax temperature must be positive")
        if self.replay_capacity < 1:
            raise ValueError("high-level replay capacity must be positive")
        if self.batch_size < 1:
            raise ValueError("high-level Q batch size must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("high-level Q learning rate must be positive")
        if (
            not np.isfinite(self.high_level_q_replay_ratio)
            or self.high_level_q_replay_ratio < 0
        ):
            raise ValueError("high-level Q replay ratio must be non-negative")
        if not self.batch_size <= self.min_replay_size <= self.replay_capacity:
            raise ValueError(
                "high-level minimum replay size must be between batch size and capacity"
            )
        if self.goal_timeout < 1:
            raise ValueError("high-level goal timeout must be positive")


class HighLevelTransitionBatch(NamedTuple):
    observation: jax.Array
    completed: jax.Array
    current_node: jax.Array
    goal: jax.Array
    reward: jax.Array
    next_observation: jax.Array
    next_completed: jax.Array
    next_current_node: jax.Array
    duration: jax.Array
    done: jax.Array
    success: jax.Array


@struct.dataclass
class HighLevelReplayBuffer:
    """Fixed-size device replay storing raw high-level transitions."""

    observation: jax.Array
    completed: jax.Array
    current_node: jax.Array
    goal: jax.Array
    reward: jax.Array
    next_observation: jax.Array
    next_completed: jax.Array
    next_current_node: jax.Array
    duration: jax.Array
    done: jax.Array
    success: jax.Array
    position: jax.Array
    size: jax.Array

    @classmethod
    def create(cls, capacity, observation_shape, num_objectives):
        observation_buffer_shape = (capacity,) + tuple(observation_shape)
        return cls(
            observation=jnp.zeros(observation_buffer_shape, dtype=jnp.float16),
            completed=jnp.zeros((capacity, num_objectives), dtype=bool),
            current_node=jnp.zeros((capacity,), dtype=jnp.int32),
            goal=jnp.zeros((capacity,), dtype=jnp.int32),
            reward=jnp.zeros((capacity,), dtype=jnp.float32),
            next_observation=jnp.zeros(
                observation_buffer_shape, dtype=jnp.float16
            ),
            next_completed=jnp.zeros((capacity, num_objectives), dtype=bool),
            next_current_node=jnp.zeros((capacity,), dtype=jnp.int32),
            duration=jnp.zeros((capacity,), dtype=jnp.int32),
            done=jnp.zeros((capacity,), dtype=bool),
            success=jnp.zeros((capacity,), dtype=bool),
            position=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
        )

    @property
    def capacity(self):
        return self.observation.shape[0]

    def add(self, transition: HighLevelTransitionBatch, valid):
        valid = valid.astype(bool)
        offsets = jnp.cumsum(valid.astype(jnp.int32)) - 1
        count = valid.astype(jnp.int32).sum()
        first_kept = jnp.maximum(count - self.capacity, 0)
        keep = valid & (offsets >= first_kept)
        positions = (self.position + offsets) % self.capacity
        positions = jnp.where(keep, positions, self.capacity)

        def append(array, values):
            return array.at[positions].set(
                values.astype(array.dtype), mode="drop"
            )

        return self.replace(
            observation=append(self.observation, transition.observation),
            completed=append(self.completed, transition.completed),
            current_node=append(self.current_node, transition.current_node),
            goal=append(self.goal, transition.goal),
            reward=append(self.reward, transition.reward),
            next_observation=append(
                self.next_observation, transition.next_observation
            ),
            next_completed=append(
                self.next_completed, transition.next_completed
            ),
            next_current_node=append(
                self.next_current_node, transition.next_current_node
            ),
            duration=append(self.duration, transition.duration),
            done=append(self.done, transition.done),
            success=append(self.success, transition.success),
            position=(self.position + count) % self.capacity,
            size=jnp.minimum(self.size + count, self.capacity),
        )

    def sample(self, key, batch_size):
        indices = jax.random.randint(
            key, (batch_size,), 0, jnp.maximum(self.size, 1)
        )
        return HighLevelTransitionBatch(
            observation=self.observation[indices].astype(jnp.float32),
            completed=self.completed[indices],
            current_node=self.current_node[indices],
            goal=self.goal[indices],
            reward=self.reward[indices],
            next_observation=self.next_observation[indices].astype(jnp.float32),
            next_completed=self.next_completed[indices],
            next_current_node=self.next_current_node[indices],
            duration=self.duration[indices],
            done=self.done[indices],
            success=self.success[indices],
        )


class HighLevelQNetwork(nn.Module):
    """Shared architecture for independent full and internal-state-free Qs."""

    num_objectives: int
    hidden_size: int
    include_internal_state: bool

    @nn.compact
    def __call__(
        self, external_state, completed, current_node, internal_state=None
    ):
        features = [
            external_state.astype(jnp.float32),
            completed.astype(jnp.float32),
            jax.nn.one_hot(current_node, self.num_objectives + 1),
        ]
        if self.include_internal_state:
            if internal_state is None:
                raise ValueError("full Q requires the symbolic internal state")
            features.insert(1, internal_state.astype(jnp.float32))
        x = jnp.concatenate(features, axis=-1)
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        return nn.Dense(self.num_objectives)(x)


@struct.dataclass
class HighLevelQState:
    full_q: TrainState
    base_q: TrainState
    target_full_params: object


def initialize_high_level_q(
    key, observation, completed, current_node, num_objectives, hidden_size, learning_rate
):
    external_state, internal_state = split_symbolic_state(observation)
    full_network = HighLevelQNetwork(num_objectives, hidden_size, True)
    base_network = HighLevelQNetwork(num_objectives, hidden_size, False)
    full_key, base_key = jax.random.split(key)
    full_params = full_network.init(
        full_key, external_state, completed, current_node, internal_state
    )
    base_params = base_network.init(base_key, external_state, completed, current_node)
    full_q = TrainState.create(
        apply_fn=full_network.apply,
        params=full_params,
        tx=optax.adam(learning_rate),
    )
    base_q = TrainState.create(
        apply_fn=base_network.apply,
        params=base_params,
        tx=optax.adam(learning_rate),
    )
    return HighLevelQState(full_q, base_q, full_params)


def evaluate_high_level_q(q_state, observation, completed, current_node):
    external_state, internal_state = split_symbolic_state(observation)
    full_values = q_state.full_q.apply_fn(
        q_state.full_q.params,
        external_state,
        completed,
        current_node,
        internal_state,
    )
    base_values = q_state.base_q.apply_fn(
        q_state.base_q.params, external_state, completed, current_node
    )
    return full_values, base_values


def get_goal_candidates(
    completed,
    graph,
    include_new_tasks=True,
    *,
    allow_unready_goals=False,
    allow_completed_goals=False,
    warmup=False,
):
    """Apply prerequisite and episode-completion eligibility independently.

    Warmup always enforces prerequisites. The legacy empty-set fallback is
    retained: prefer eligible goals, or all graph nodes if all are completed.
    """

    cluster_count = graph.shape[0]
    cluster_completed = completed[..., :cluster_count]
    prerequisites_met = jnp.all(
        ~graph.astype(bool) | cluster_completed[..., :, None], axis=-2
    )
    remaining = (
        jnp.ones_like(cluster_completed) if allow_completed_goals else ~cluster_completed
    )
    prerequisites_allowed = prerequisites_met | (
        allow_unready_goals & ~jnp.asarray(warmup)
    )
    ready = prerequisites_allowed & remaining
    has_ready = jnp.any(ready, axis=-1, keepdims=True)
    has_remaining = jnp.any(remaining, axis=-1, keepdims=True)
    fallback = jnp.where(has_remaining, remaining, jnp.ones_like(remaining))
    candidates = jnp.where(has_ready, ready, fallback)
    if include_new_tasks:
        candidates = jnp.concatenate(
            [candidates, jnp.ones(candidates.shape[:-1] + (1,), dtype=bool)],
            axis=-1,
        )
    return candidates


def get_ready_goal_candidates(completed, graph, include_new_tasks=True):
    """Backward-compatible ready, unfinished candidate policy."""
    return get_goal_candidates(completed, graph, include_new_tasks)


def compute_q_residual(full_q, base_q):
    return full_q - base_q


def compute_dynamic_edge_scores(
    graph, current_node, full_q, base_q, residual_weight, *,
    structural_prior_weight=1.0, score_source="residual",
):
    """Return lambda*A + beta*signal, preserving the legacy beta argument name.

    Only the selection logits change. Q learning and candidate eligibility are
    independent of lambda and the selected signal.
    """
    cluster_count = graph.shape[0]
    safe_node = jnp.clip(current_node, 0, max(cluster_count - 1, 0))
    static_scores = graph[safe_node].astype(jnp.float32)
    valid_node = current_node < cluster_count
    static_scores = jnp.where(valid_node[..., None], static_scores, 0.0)
    if full_q.shape[-1] > cluster_count:
        padding = full_q.shape[-1] - cluster_count
        pad_width = (
            [(0, 0)] * (static_scores.ndim - 1) + [(0, padding)]
        )
        static_scores = jnp.pad(static_scores, pad_width)
    if score_source == "residual":
        signal = compute_q_residual(full_q, base_q)
    elif score_source == "base-q":
        signal = base_q
    elif score_source == "full-q":
        signal = full_q
    else:
        raise ValueError("dynamic score source must be residual, base-q, or full-q")
    return structural_prior_weight * static_scores + residual_weight * signal


def stable_masked_softmax(scores, candidates, temperature=1.0):
    scaled = scores / temperature
    masked = jnp.where(candidates, scaled, -jnp.inf)
    maximum = jnp.max(masked, axis=-1, keepdims=True)
    weights = jnp.where(candidates, jnp.exp(masked - maximum), 0.0)
    return weights / weights.sum(axis=-1, keepdims=True)


def select_high_level_goal(
    key, candidates, dynamic_scores, temperature, use_uniform
):
    uniform_scores = jnp.zeros_like(dynamic_scores)
    scores = jnp.where(use_uniform, uniform_scores, dynamic_scores)
    probabilities = stable_masked_softmax(scores, candidates, temperature)
    selected = jax.random.categorical(key, jnp.log(probabilities))
    return selected.astype(jnp.int32), probabilities


def compute_smdp_target(reward, duration, done, next_q, candidates, gamma):
    masked_next_q = jnp.where(candidates, next_q, -jnp.inf)
    next_value = jnp.max(masked_next_q, axis=-1)
    bootstrap = jnp.where(done, 0.0, next_value)
    return reward + (gamma ** duration) * bootstrap


def compute_q_update_count(
    new_transition_count, batch_size, high_level_q_replay_ratio
):
    """Return ceil(rho * newly collected high-level transitions / batch size)."""

    replay_samples = (
        jnp.asarray(new_transition_count, dtype=jnp.float32)
        * high_level_q_replay_ratio
    )
    return jnp.ceil(replay_samples / batch_size).astype(jnp.int32)


def high_level_option_done(goal_done, duration, env_done, goal_timeout):
    """End an option only on target success, timeout, or episode termination."""

    return goal_done | (duration >= goal_timeout) | env_done


def base_q_regression_loss(base_values, full_target_values):
    return jnp.square(base_values - jax.lax.stop_gradient(full_target_values)).mean()


def update_high_level_q(
    q_state, batch, graph, include_new_tasks, gamma, *,
    allow_unready_goals=False, allow_completed_goals=False, warmup=False,
):
    external_state, internal_state = split_symbolic_state(batch.observation)
    next_external_state, next_internal_state = split_symbolic_state(
        batch.next_observation
    )
    next_full_values = q_state.full_q.apply_fn(
        q_state.target_full_params,
        next_external_state,
        batch.next_completed,
        batch.next_current_node,
        next_internal_state,
    )
    next_candidates = get_goal_candidates(
        batch.next_completed, graph, include_new_tasks,
        allow_unready_goals=allow_unready_goals,
        allow_completed_goals=allow_completed_goals,
        warmup=warmup,
    )
    targets = jax.lax.stop_gradient(
        compute_smdp_target(
            batch.reward,
            batch.duration,
            batch.done,
            next_full_values,
            next_candidates,
            gamma,
        )
    )

    def full_loss(parameters):
        values = q_state.full_q.apply_fn(
            parameters,
            external_state,
            batch.completed,
            batch.current_node,
            internal_state,
        )
        selected = jnp.take_along_axis(values, batch.goal[:, None], axis=-1)[:, 0]
        return jnp.square(selected - targets).mean(), selected

    (full_loss_value, full_values), full_gradients = jax.value_and_grad(
        full_loss, has_aux=True
    )(q_state.full_q.params)
    full_q = q_state.full_q.apply_gradients(grads=full_gradients)

    target_full_values = q_state.full_q.apply_fn(
        q_state.target_full_params,
        external_state,
        batch.completed,
        batch.current_node,
        internal_state,
    )
    target_full_values = jnp.take_along_axis(
        target_full_values, batch.goal[:, None], axis=-1
    )[:, 0]

    def base_loss(parameters):
        values = q_state.base_q.apply_fn(
            parameters,
            external_state,
            batch.completed,
            batch.current_node,
        )
        selected = jnp.take_along_axis(values, batch.goal[:, None], axis=-1)[:, 0]
        return base_q_regression_loss(selected, target_full_values), selected

    (base_loss_value, base_values), base_gradients = jax.value_and_grad(
        base_loss, has_aux=True
    )(q_state.base_q.params)
    base_q = q_state.base_q.apply_gradients(grads=base_gradients)
    residual = compute_q_residual(full_values, base_values)
    metrics = {
        "full_q_loss": full_loss_value,
        "base_q_loss": base_loss_value,
        "full_q_mean": full_values.mean(),
        "full_q_std": full_values.std(),
        "base_q_mean": base_values.mean(),
        "base_q_std": base_values.std(),
        "q_residual_mean": residual.mean(),
        "q_residual_std": residual.std(),
    }
    return HighLevelQState(full_q, base_q, full_q.params), metrics


__all__ = [
    "DynamicGraphConfig",
    "HighLevelQNetwork",
    "HighLevelQState",
    "HighLevelReplayBuffer",
    "HighLevelTransitionBatch",
    "INTERNAL_STATE_START",
    "INTERNAL_STATE_STOP",
    "SYMBOLIC_OBSERVATION_SIZE",
    "base_q_regression_loss",
    "compute_dynamic_edge_scores",
    "compute_q_residual",
    "compute_q_update_count",
    "compute_smdp_target",
    "evaluate_high_level_q",
    "get_ready_goal_candidates",
    "get_goal_candidates",
    "high_level_option_done",
    "initialize_high_level_q",
    "select_high_level_goal",
    "split_symbolic_state",
    "stable_masked_softmax",
    "update_high_level_q",
]
