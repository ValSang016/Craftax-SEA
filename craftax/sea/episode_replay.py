"""Complete episode event replay, independent of prediction-data subsampling."""
from collections import deque
from functools import lru_cache
import math

import jax
import jax.numpy as jnp
import numpy as np

from craftax.sea.discovery import TransitionDataset
from craftax.sea.networks import ActorCriticRNN, ScannedGRU
from craftax.sea.vector_env import SeaAutoResetVecEnv


class EpisodeEventBuffer:
    """Retain all events of the latest completed episodes with >=2 events."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("episode capacity must be positive")
        self.pending = {}
        self.completed = deque(maxlen=capacity)
        self.completed_count = 0

    def append(self, episode_id, event, done):
        if event is not None:
            self.pending.setdefault(episode_id, []).append(event)
        if done:
            events = self.pending.pop(episode_id, [])
            if len(events) >= 2:
                self.completed.append(events)
                self.completed_count += 1

    def dataset(self):
        rows = [row for episode in self.completed for row in episode]
        if not rows:
            raise ValueError("no complete episodes with at least two achievements; increase collection window")
        fields = [np.stack(values) for values in zip(*rows)]
        return TransitionDataset(
            observation=fields[0], action=fields[1].astype(np.int16),
            next_observation=fields[2], terminal=fields[3].astype(bool),
            event_count=fields[4].astype(np.float32),
            contrast_eligible=np.ones(len(rows), dtype=bool),
            episode_id=fields[5].astype(np.int32), episode_step=fields[6].astype(np.int32),
            new_achievements=fields[7].astype(bool),
        )


def collect_episode_event_dataset(env, policy_params, ppo_config, *,
                                  interactions=1_000_000, episode_capacity=256,
                                  pixel=False, seed=0, chunk_steps=8):
    """Replay the collection-policy stream and retain complete event groups.

    The PRNG schedule matches collect_transition_dataset. No event sampling is
    performed. Only episodes ending within the window enter the FIFO buffer;
    unfinished episodes are discarded. Float observations are sent to host
    in small chunks, and only positive transitions are retained there.
    """
    if interactions < ppo_config.num_envs:
        raise ValueError("interactions must be at least num_envs")
    vec = SeaAutoResetVecEnv(env, ppo_config.num_envs, ppo_config.reset_ratio)
    params = env.default_params
    network = ActorCriticRNN(action_dim=env.action_space(params).n,
                            hidden_size=ppo_config.hidden_size, pixel=pixel,
                            num_objectives=0)
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    observation, state = vec.reset(reset_key, params)
    carry = (state, observation, jnp.zeros(ppo_config.num_envs, dtype=bool),
             ScannedGRU.initialize_carry(ppo_config.num_envs, ppo_config.hidden_size),
             jnp.zeros(ppo_config.num_envs, dtype=jnp.int32), key)

    def compact(obs):
        # Preserve the original float observation output and perform the same
        # host-side conversion as prediction collection, avoiding fused rounding.
        return obs

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
                    jnp.zeros((1, ppo_config.num_envs, 1), dtype=bool))
                action = jax.random.categorical(action_key, logits[0])
                episode_id = jnp.arange(ppo_config.num_envs) + episode_count * ppo_config.num_envs
                next_obs, next_state, reward, done, info = vec.step(step_key, state, action, params)
                rows = (compact(obs), action, compact(info['terminal_obs']), done,
                        reward, episode_id, state.env_state.timestep, info['new_achievements'])
                return (next_state, next_obs, done, hidden,
                        episode_count + done.astype(jnp.int32), key), rows
            return jax.lax.scan(step, carry, None, length=length)
        return collect

    buffer = EpisodeEventBuffer(episode_capacity)
    steps = math.ceil(interactions / ppo_config.num_envs)
    for start in range(0, steps, chunk_steps):
        length = min(chunk_steps, steps-start)
        carry, chunk = make_chunk(length)(carry)
        chunk = tuple(np.asarray(x) for x in chunk)
        interesting = (chunk[4] > 0) | chunk[3]
        for t, e in np.argwhere(interesting):
            if (start + t) * ppo_config.num_envs + e >= interactions:
                continue
            # Copy event slices so pending episodes do not retain full chunks.
            event = None
            if chunk[4][t, e] > 0:
                event = [x[t, e].copy() for x in chunk]
                for index in (0, 2):
                    event[index] = (np.rint(event[index] * 255).astype(np.uint8) if pixel
                                    else event[index].astype(np.float16))
                event = tuple(event)
            buffer.append(int(chunk[5][t, e]), event, bool(chunk[3][t, e]))
        if start % (chunk_steps * 16) == 0 or start + length == steps:
            print(f"[episode replay] {min((start+length)*ppo_config.num_envs, interactions)}/{interactions} "
                  f"interactions; completed groups={buffer.completed_count}, retained={len(buffer.completed)}", flush=True)
    return buffer.dataset()
