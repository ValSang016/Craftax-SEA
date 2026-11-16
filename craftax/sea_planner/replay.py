"""High-level replay of finished options for the supervised option model.

One sample is one option: ``(obs, completed, current_node, goal)`` at the
moment the goal was given, and its outcome ``success, duration, died`` with
the situation ``(next_obs, next_completed, next_current_node)`` at its end.
No returns are stored: the option model is trained by supervision only.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct


class OptionTransition(NamedTuple):
    observation: jax.Array
    completed: jax.Array
    current_node: jax.Array
    goal: jax.Array
    next_observation: jax.Array
    next_completed: jax.Array
    next_current_node: jax.Array
    duration: jax.Array
    done: jax.Array
    success: jax.Array
    died: jax.Array


@struct.dataclass
class OptionReplayBuffer:
    """Fixed-size device ring buffer (observations stored as float16)."""

    observation: jax.Array
    completed: jax.Array
    current_node: jax.Array
    goal: jax.Array
    next_observation: jax.Array
    next_completed: jax.Array
    next_current_node: jax.Array
    duration: jax.Array
    done: jax.Array
    success: jax.Array
    died: jax.Array
    position: jax.Array
    size: jax.Array

    @classmethod
    def create(cls, capacity, observation_shape, num_goals):
        observation_shape = (capacity,) + tuple(observation_shape)
        return cls(
            observation=jnp.zeros(observation_shape, dtype=jnp.float16),
            completed=jnp.zeros((capacity, num_goals), dtype=bool),
            current_node=jnp.zeros((capacity,), dtype=jnp.int32),
            goal=jnp.zeros((capacity,), dtype=jnp.int32),
            next_observation=jnp.zeros(observation_shape, dtype=jnp.float16),
            next_completed=jnp.zeros((capacity, num_goals), dtype=bool),
            next_current_node=jnp.zeros((capacity,), dtype=jnp.int32),
            duration=jnp.zeros((capacity,), dtype=jnp.int32),
            done=jnp.zeros((capacity,), dtype=bool),
            success=jnp.zeros((capacity,), dtype=bool),
            died=jnp.zeros((capacity,), dtype=bool),
            position=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
        )

    @property
    def capacity(self):
        return self.observation.shape[0]

    def add(self, transition: OptionTransition, valid):
        """Append the rows where ``valid`` is true, keeping the newest ones."""
        valid = valid.astype(bool)
        offsets = jnp.cumsum(valid.astype(jnp.int32)) - 1
        count = valid.astype(jnp.int32).sum()
        first_kept = jnp.maximum(count - self.capacity, 0)
        keep = valid & (offsets >= first_kept)
        positions = (self.position + offsets) % self.capacity
        positions = jnp.where(keep, positions, self.capacity)

        def append(array, values):
            values = jnp.broadcast_to(jnp.asarray(values), valid.shape + array.shape[1:])
            return array.at[positions].set(values.astype(array.dtype), mode="drop")

        stored = {
            name: append(getattr(self, name), getattr(transition, name))
            for name in OptionTransition._fields
        }
        return self.replace(
            **stored,
            position=(self.position + count) % self.capacity,
            size=jnp.minimum(self.size + count, self.capacity),
        )

    def sample(self, key, batch_size):
        indices = jax.random.randint(key, (batch_size,), 0, jnp.maximum(self.size, 1))
        rows = {name: getattr(self, name)[indices] for name in OptionTransition._fields}
        rows["observation"] = rows["observation"].astype(jnp.float32)
        rows["next_observation"] = rows["next_observation"].astype(jnp.float32)
        return OptionTransition(**rows)


def option_update_count(recorded_count, batch_size, replay_ratio):
    """round(recorded options * replay ratio / batch size) supervised updates.

    Counts options actually written to replay (censored warmup options are
    not), not every option ending.
    """
    samples = jnp.asarray(recorded_count, dtype=jnp.float32) * replay_ratio
    return jnp.round(samples / batch_size).astype(jnp.int32)


__all__ = ["OptionReplayBuffer", "OptionTransition", "option_update_count"]
