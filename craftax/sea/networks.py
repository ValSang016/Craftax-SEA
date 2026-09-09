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
        x = nn.Conv(32, (5, 5), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        return x.reshape(leading_shape + (self.hidden_size,))


class ActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 512
    pixel: bool = False
    num_objectives: int = 0
    goal_embedding_size: int = 32
    include_completed: bool = True

    @nn.compact
    def __call__(self, hidden, observation, reset, objective, completed):
        embedding = ObservationEncoder(self.hidden_size, self.pixel)(observation)

        if self.num_objectives > 0:
            goal_embedding = nn.Embed(
                num_embeddings=self.num_objectives,
                features=self.goal_embedding_size,
            )(objective.astype(jnp.int32))
            features = [embedding, goal_embedding]
            if self.include_completed:
                features.append(completed.astype(jnp.float32))
            embedding = jnp.concatenate(features, axis=-1)
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
    embedding_size: int = 128
    pixel: bool = False

    @nn.compact
    def __call__(self, observation, action, next_observation, terminal):
        encoder = ObservationEncoder(self.hidden_size, self.pixel)
        before = encoder(observation)
        after = encoder(next_observation)
        half = self.hidden_size // 2
        # Unlike TorchBeast, the SEA vector wrapper retains the true terminal
        # observation instead of replacing it with the next reset observation,
        # so both sides of a terminal transition remain valid features.
        state_features = jnp.concatenate(
            [before[..., :half], after[..., half:]],
            axis=-1,
        )
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate(
            [state_features, action_one_hot, terminal[..., None]], axis=-1
        )
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x)) + state_features
        embedding = nn.Dense(self.embedding_size)(x)
        reward_logit = nn.Dense(1)(x)[..., 0]
        return reward_logit, embedding


__all__ = [
    "ActorCriticRNN",
    "ObservationEncoder",
    "ScannedGRU",
    "TransitionEncoder",
]
