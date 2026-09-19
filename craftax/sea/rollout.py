"""Policy rollout utilities for SEA discovery data collection."""

from __future__ import annotations

import math
from functools import lru_cache

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
    contrast_interactions: int = 1_000_000,
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
        jnp.int32(0),
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
        # Several vector environments can unlock achievements on the same
        # step.  Retaining only one of them capped a 1M/1024 contrast window at
        # 977 positives despite a requested limit of 100k.
        positive_steps = max(
            1,
            math.ceil(
                min(interactions, contrast_interactions) / ppo_config.num_envs
            ),
        )
        positive_per_step = min(
            ppo_config.num_envs,
            max(1, math.ceil(max_positive_transitions / positive_steps)),
        )
        positive_acceptance = min(
            1.0,
            max_positive_transitions / (positive_steps * positive_per_step),
        )
    else:
        random_per_step = ppo_config.num_envs
        random_acceptance = 1.0
        positive_per_step = 1
        positive_acceptance = 0.0

    @lru_cache(maxsize=2)
    def make_chunk(length):
        @jax.jit
        def collect(carry):
            def step(carry, _):
                (
                    env_state,
                    observation,
                    last_done,
                    hidden,
                    episode_count,
                    rollout_step,
                    key,
                ) = carry
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
                contrast_eligible = jnp.full(
                    (ppo_config.num_envs,),
                    rollout_step * ppo_config.num_envs < contrast_interactions,
                    dtype=bool,
                )
                full_sample = (
                    observation,
                    action,
                    info["terminal_obs"],
                    done,
                    reward,
                    contrast_eligible,
                    episode_id,
                    episode_step,
                    info["new_achievements"],
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
                    event_mask = (reward > 0) & contrast_eligible
                    scores = jnp.where(
                        event_mask,
                        jax.random.uniform(positive_key, (ppo_config.num_envs,)),
                        -jnp.ones((ppo_config.num_envs,)),
                    )
                    positive_scores, positive_indices = jax.lax.top_k(
                        scores, positive_per_step
                    )
                    positive_valid = (positive_scores >= 0.0) & (
                        jax.random.uniform(sample_key, (positive_per_step,))
                        < positive_acceptance
                    )

                    def take(values, indices):
                        return values[indices]

                    sample = (
                        tuple(take(values, random_indices) for values in full_sample),
                        tuple(take(values, positive_indices) for values in full_sample),
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
                    rollout_step + 1,
                    key,
                )
                return carry, sample

            return jax.lax.scan(step, carry, None, length=length)

        return collect

    arrays = [[] for _ in range(9)]
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
                positive_values = np.asarray(positive_chunk[index]).reshape(
                    (-1,) + positive_chunk[index].shape[2:]
                )
                destination.append(random_values[random_valid])
                destination.append(positive_values[positive_valid])
        else:
            for destination, values in zip(arrays, chunk):
                destination.append(np.asarray(values).reshape((-1,) + values.shape[2:]))
        collected_steps += length
        if collected_steps % (chunk_steps * 32) == 0 or collected_steps == steps_per_env:
            print(f"[collection] {min(collected_steps * ppo_config.num_envs, interactions)}/{interactions}", flush=True)

    flattened = [np.concatenate(values, axis=0) for values in arrays]
    if not sampled:
        flattened = [values[:interactions] for values in flattened]
    elif len(flattened[1]):
        # A positive transition can also be selected by the uniform sampler.
        # SEA's determinant objective treats repetitions within an episode as
        # conflicts, so deduplicate by the stable (episode, step) identity.
        identities = np.stack((flattened[6], flattened[7]), axis=1)
        _, unique_indices = np.unique(identities, axis=0, return_index=True)
        unique_indices.sort()
        flattened = [values[unique_indices] for values in flattened]
    (
        observation,
        action,
        next_observation,
        terminal,
        reward,
        contrast_eligible,
        episode_id,
        episode_step,
        new_achievements,
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
        contrast_eligible=contrast_eligible.astype(bool),
        new_achievements=new_achievements.astype(bool),
    )


__all__ = ["collect_transition_dataset", "iter_policy_transitions"]


def iter_policy_transitions(
    env, policy_params, ppo_config, *, interactions, pixel=False, seed=0,
    chunk_steps=80, network_class=ActorCriticRNN,
):
    """Yield fresh, consecutive rollout chunks without sampling or replay.

    Carry both environment and policy RNN state across chunks. Every yielded
    transition is unique within this stream, including a truncated final chunk.
    Keep true terminal observations and compact frames on the host as in the
    diagnostic collector. At most one rollout chunk is transferred at a time.
    """
    if interactions <= 0 or chunk_steps <= 0:
        raise ValueError("interactions and chunk_steps must be positive")
    vec = SeaAutoResetVecEnv(env, ppo_config.num_envs, ppo_config.reset_ratio)
    params = env.default_params
    network = network_class(
        action_dim=env.action_space(params).n, hidden_size=ppo_config.hidden_size,
        pixel=pixel, num_objectives=0,
    )
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    observation, env_state = vec.reset(reset_key, params)
    carry = (
        env_state, observation, jnp.zeros(ppo_config.num_envs, dtype=bool),
        ScannedGRU.initialize_carry(ppo_config.num_envs, ppo_config.hidden_size),
        jnp.zeros(ppo_config.num_envs, dtype=jnp.int32), key,
    )

    @lru_cache(maxsize=2)
    def make_chunk(length):
        @jax.jit
        def collect(carry):
            def step(carry, _):
                state, obs, last_done, hidden, episode_count, key = carry
                key, action_key, step_key, _ = jax.random.split(key, 4)
                hidden, logits, _ = network.apply(
                    policy_params, hidden, obs[None], last_done[None],
                    jnp.zeros((1, ppo_config.num_envs), dtype=jnp.int32),
                    jnp.zeros((1, ppo_config.num_envs, 1), dtype=bool),
                )
                action = jax.random.categorical(action_key, logits[0])
                episode_id = jnp.arange(ppo_config.num_envs) + episode_count * ppo_config.num_envs
                next_obs, next_state, reward, done, info = vec.step(step_key, state, action, params)
                row = (obs, action, info['terminal_obs'], done, reward,
                       episode_id, state.env_state.timestep, info['new_achievements'])
                return (next_state, next_obs, done, hidden,
                        episode_count + done.astype(jnp.int32), key), row
            return jax.lax.scan(step, carry, None, length=length)
        return collect

    consumed = 0
    while consumed < interactions:
        length = min(chunk_steps, math.ceil((interactions-consumed)/ppo_config.num_envs))
        carry, chunk = make_chunk(length)(carry)
        size = min(length * ppo_config.num_envs, interactions-consumed)
        arrays = [np.asarray(x).reshape((-1,)+x.shape[2:])[:size] for x in chunk]
        for index in (0, 2):
            arrays[index] = (np.rint(arrays[index]*255).astype(np.uint8) if pixel
                             else arrays[index].astype(np.float16))
        yield TransitionDataset(
            observation=arrays[0], action=arrays[1].astype(np.int16),
            next_observation=arrays[2], terminal=arrays[3].astype(bool),
            event_count=arrays[4].astype(np.float32),
            episode_id=arrays[5].astype(np.int32), episode_step=arrays[6].astype(np.int32),
            contrast_eligible=np.ones(size, dtype=bool), new_achievements=arrays[7].astype(bool),
        )
        consumed += size
