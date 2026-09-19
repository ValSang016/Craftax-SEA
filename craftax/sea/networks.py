"""Flax networks used by PPO-RNN and SEA transition discovery."""

from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ScannedGRU(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, inputs):
        embedding, reset = inputs
        zero_carry = jnp.zeros_like(carry)
        carry = jnp.where(reset[:, None], zero_carry, carry)
        carry, output = nn.GRUCell(features=self.hidden_size)(carry, embedding)
        return carry, output

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


class ObservationEncoder(nn.Module):
    hidden_size: int
    pixel: bool = False

    @nn.compact
    def __call__(self, observation):
        was_uint8 = observation.dtype == jnp.uint8
        observation = observation.astype(jnp.float32)
        if not self.pixel:
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(observation)
            return nn.relu(x)

        leading_shape = observation.shape[:-3]
        if was_uint8:
            observation = observation / 255.0
        x = observation.reshape((-1,) + observation.shape[-3:])
        # Match SEA's IMPALA/Crafter image torso.  Craftax uses 63x63 RGB
        # frames instead of Crafter's 64x64 frames, but these kernels remain
        # valid and produce the same 4x4 final spatial extent.
        x = nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        return x.reshape(leading_shape + (self.hidden_size,))


class ActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 256
    pixel: bool = False
    num_objectives: int = 0
    include_completed: bool = False

    @nn.compact
    def __call__(self, hidden, observation, reset, objective, completed):
        embedding = ObservationEncoder(self.hidden_size, self.pixel)(observation)

        if self.num_objectives > 0:
            objective_one_hot = jax.nn.one_hot(objective, self.num_objectives)
            features = [embedding, objective_one_hot]
            if self.include_completed:
                features.append(completed.astype(jnp.float32))
            embedding = jnp.concatenate(features, axis=-1)
            embedding = nn.relu(nn.Dense(self.hidden_size)(embedding))
            embedding = nn.relu(nn.Dense(self.hidden_size)(embedding))

        hidden, embedding = ScannedGRU(self.hidden_size)(hidden, (embedding, reset))

        actor = nn.relu(
            nn.Dense(self.hidden_size, kernel_init=orthogonal(2))(embedding)
        )
        actor = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(2))(actor))
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)

        critic = nn.relu(
            nn.Dense(self.hidden_size, kernel_init=orthogonal(2))(embedding)
        )
        critic = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(2))(critic))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return hidden, logits, jnp.squeeze(value, axis=-1)


class TransitionEncoder(nn.Module):
    """Encode ``(s_t, a_t, s_{t+1})`` and predict reward occurrence."""

    action_dim: int = 17
    hidden_size: int = 256
    embedding_size: int = 256
    pixel: bool = False

    @nn.compact
    def __call__(self, observation, action, next_observation, terminal):
        encoder = ObservationEncoder(self.hidden_size, self.pixel)
        before = encoder(observation)
        after = encoder(next_observation)
        half = self.hidden_size // 2
        # Original SEA masks the next-state half on episode termination.  The
        # JAX wrapper still retains the true terminal observation for metrics
        # and non-terminal event classification.
        after_half = after[..., :half] * (~terminal)[..., None]
        state_features = jnp.concatenate(
            [after_half, before[..., half:]],
            axis=-1,
        )
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate(
            [state_features, action_one_hot, terminal[..., None]], axis=-1
        )
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x)) + state_features
        # SEA clusters the residual transition feature directly.  Keep an
        # optional projection only for explicitly requested non-256 variants.
        embedding = x if self.embedding_size == self.hidden_size else nn.Dense(
            self.embedding_size
        )(x)
        reward_logit = nn.Dense(1)(x)[..., 0]
        return reward_logit, embedding


__all__ = [
    "ActorCriticRNN",
    "ObservationEncoder",
    "ScannedGRU",
    "TransitionEncoder",
]
