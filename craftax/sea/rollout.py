"""Policy rollout utilities for SEA discovery data collection."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from craftax.sea.discovery import TransitionDataset
from craftax.sea.networks import ActorCriticRNN, ScannedGRU
from craftax.sea.ppo_rnn import categorical_log_prob
from craftax.sea.vector_env import SeaAutoResetVecEnv


def collect_transition_dataset(
    env,
    policy_params,
    ppo_config,
    *,
    interactions: int,
    pixel: bool = False,
    seed: int = 0,
    chunk_steps: int = 64,
    max_random_transitions: int | None = 200_000,
    max_positive_transitions: int = 100_000,
):
    """Collect a bounded host dataset with a frozen PPO-RNN policy.

    The environment still runs for ``interactions`` transitions, but only a
    uniform random subset and a separate sample of achievement transitions are
    transferred to the host.  This keeps a 50M-step pixel rollout bounded.
    Pass ``max_random_transitions=None`` to retain every transition (suitable
    only for small diagnostics).  Pixel frames are stored as uint8 and symbolic
    observations as float16.
    """

    if interactions < ppo_config.num_envs:
        raise ValueError("interactions must be at least num_envs")
    vec_env = SeaAutoResetVecEnv(env, ppo_config.num_envs, ppo_config.reset_ratio)
    params = env.default_params
    network = ActorCriticRNN(
        action_dim=env.action_space(params).n,
        hidden_size=ppo_config.hidden_size,
        pixel=pixel,
        num_objectives=0,
    )
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    observation, env_state = vec_env.reset(reset_key, params)
    carry = (
        env_state,
        observation,
        jnp.zeros((ppo_config.num_envs,), dtype=bool),
        ScannedGRU.initialize_carry(ppo_config.num_envs, ppo_config.hidden_size),
        jnp.zeros((ppo_config.num_envs,), dtype=jnp.int32),
        key,
    )

    steps_per_env = math.ceil(interactions / ppo_config.num_envs)
    sampled = (
        max_random_transitions is not None and max_random_transitions < interactions
    )
    if sampled:
        random_per_step = min(
            ppo_config.num_envs,
            max(1, math.ceil(max_random_transitions / steps_per_env)),
        )
        random_acceptance = min(
            1.0, max_random_transitions / (steps_per_env * random_per_step)
        )
        positive_acceptance = min(1.0, max_positive_transitions / steps_per_env)
    else:
        random_per_step = ppo_config.num_envs
        random_acceptance = 1.0
        positive_acceptance = 0.0

    def make_chunk(length):
        @jax.jit
        def collect(carry):
            def step(carry, _):
                env_state, observation, last_done, hidden, episode_count, key = carry
                key, action_key, step_key, sample_key = jax.random.split(key, 4)
                zeros_goal = jnp.zeros((1, ppo_config.num_envs), dtype=jnp.int32)
                zeros_completed = jnp.zeros((1, ppo_config.num_envs, 1), dtype=bool)
                hidden, logits, _ = network.apply(
                    policy_params,
                    hidden,
                    observation[None],
                    last_done[None],
                    zeros_goal,
                    zeros_completed,
                )
                action = jax.random.categorical(action_key, logits[0])
                # Force evaluation of the same sampling path as PPO, including
                # its log-softmax, while only storing the action here.
                _ = categorical_log_prob(logits[0], action)
                episode_id = (
                    jnp.arange(ppo_config.num_envs, dtype=jnp.int32)
                    + episode_count * ppo_config.num_envs
                )
                episode_step = env_state.env_state.timestep
                next_observation, next_state, reward, done, info = vec_env.step(
                    step_key, env_state, action, params
                )
                full_sample = (
                    observation,
                    action,
                    info["terminal_obs"],
                    done,
                    reward,
                    episode_id,
                    episode_step,
                )
                if sampled:
                    sample_key, index_key, accept_key, positive_key = jax.random.split(
                        sample_key, 4
                    )
                    random_indices = jax.random.choice(
                        index_key,
                        ppo_config.num_envs,
                        shape=(random_per_step,),
                        replace=False,
                    )
                    random_valid = (
                        jax.random.uniform(accept_key, (random_per_step,))
                        < random_acceptance
                    )
                    event_mask = reward > 0
                    scores = jnp.where(
                        event_mask,
                        jax.random.uniform(positive_key, (ppo_config.num_envs,)),
                        -jnp.ones((ppo_config.num_envs,)),
                    )
                    positive_index = jnp.argmax(scores)
                    positive_valid = event_mask[positive_index] & (
                        jax.random.uniform(sample_key) < positive_acceptance
                    )

                    def take(values, indices):
                        return values[indices]

                    sample = (
                        tuple(take(values, random_indices) for values in full_sample),
                        tuple(values[positive_index] for values in full_sample),
                        random_valid,
                        positive_valid,
                    )
                else:
                    sample = full_sample
                carry = (
                    next_state,
                    next_observation,
                    done,
                    hidden,
                    episode_count + done.astype(jnp.int32),
                    key,
                )
                return carry, sample

            return jax.lax.scan(step, carry, None, length=length)

        return collect

    arrays = [[] for _ in range(7)]
    collected_steps = 0
    while collected_steps < steps_per_env:
        length = min(chunk_steps, steps_per_env - collected_steps)
        carry, chunk = make_chunk(length)(carry)
        if sampled:
            random_chunk, positive_chunk, random_valid, positive_valid = chunk
            random_valid = np.asarray(random_valid).reshape(-1)
            positive_valid = np.asarray(positive_valid).reshape(-1)
            for index, destination in enumerate(arrays):
                random_values = np.asarray(random_chunk[index]).reshape(
                    (-1,) + random_chunk[index].shape[2:]
                )
                positive_values = np.asarray(positive_chunk[index])
                destination.append(random_values[random_valid])
                destination.append(positive_values[positive_valid])
        else:
            for destination, values in zip(arrays, chunk):
                destination.append(np.asarray(values).reshape((-1,) + values.shape[2:]))
        collected_steps += length

    flattened = [np.concatenate(values, axis=0) for values in arrays]
    if not sampled:
        flattened = [values[:interactions] for values in flattened]
    (
        observation,
        action,
        next_observation,
        terminal,
        reward,
        episode_id,
        episode_step,
    ) = flattened
    if pixel:
        observation = np.rint(observation * 255).astype(np.uint8)
        next_observation = np.rint(next_observation * 255).astype(np.uint8)
    else:
        observation = observation.astype(np.float16)
        next_observation = next_observation.astype(np.float16)
    return TransitionDataset(
        observation=observation,
        action=action.astype(np.int16),
        next_observation=next_observation,
        terminal=terminal.astype(bool),
        event_count=reward.astype(np.float32),
        episode_id=episode_id.astype(np.int32),
        episode_step=episode_step.astype(np.int32),
    )


__all__ = ["collect_transition_dataset"]
