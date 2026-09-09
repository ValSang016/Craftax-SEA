"""Vectorisation and efficient auto-reset for SEA Craftax environments."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _select_batch(mask, when_true, when_false):
    """Select PyTree leaves with a leading environment dimension."""

    def select_leaf(x, y):
        expanded = mask.reshape((mask.shape[0],) + (1,) * (x.ndim - 1))
        return jnp.where(expanded, x, y)

    return jax.tree.map(select_leaf, when_true, when_false)


class SeaAutoResetVecEnv:
    """Batch a no-auto-reset SEA environment and preserve terminal observations.

    Only ``num_envs // reset_ratio`` fresh worlds are generated per step.  If
    more environments finish, reset worlds are shared cyclically, matching the
    optimistic-reset idea used by the official Craftax baselines.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int = 16):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if reset_ratio < 1 or num_envs % reset_ratio:
            raise ValueError("reset_ratio must be a positive divisor of num_envs")
        self.env = env
        self.num_envs = int(num_envs)
        self.reset_ratio = int(reset_ratio)
        self.num_resets = self.num_envs // self.reset_ratio
        self._reset_many = jax.vmap(self.env.reset, in_axes=(0, None))
        self._step_many = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        params = self.env.default_params if params is None else params
        keys = jax.random.split(key, self.num_envs)
        return self._reset_many(keys, params)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        params = self.env.default_params if params is None else params
        step_key, reset_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, self.num_envs)
        terminal_obs, stepped_state, reward, done, info = self._step_many(
            step_keys, state, action, params
        )

        reset_keys = jax.random.split(reset_key, self.num_resets)
        reset_obs, reset_state = self._reset_many(reset_keys, params)

        # Finished environments are numbered in encounter order and assigned a
        # generated reset world.  Sharing only happens if more than num_resets
        # finish in the same vector step.
        reset_index = jnp.maximum(jnp.cumsum(done.astype(jnp.int32)) - 1, 0)
        reset_index = reset_index % self.num_resets
        reset_obs_for_env = reset_obs[reset_index]
        reset_state_for_env = jax.tree.map(lambda x: x[reset_index], reset_state)

        next_obs = _select_batch(done, reset_obs_for_env, terminal_obs)
        next_state = _select_batch(done, reset_state_for_env, stepped_state)
        info = dict(info)
        info["terminal_obs"] = terminal_obs
        return next_obs, next_state, reward, done, info


__all__ = ["SeaAutoResetVecEnv"]
