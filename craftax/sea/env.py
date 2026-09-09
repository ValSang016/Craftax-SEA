"""Craftax-Classic with the environment semantics used by SEA.

This module reproduces the non-vanilla Crafter settings from the SEA codebase:

* the player is immortal and health is always nine;
* lava and health do not terminate the episode;
* an episode ends after 100 steps without a newly unlocked achievement;
* reward is the number of newly unlocked achievements (no health shaping);
* creatures only take damage when the current hit is lethal; and
* cow, skeleton and zombie health are 2, 3 and 5 respectively.

The environment never auto-resets.  This is important for SEA because the
transition encoder must see the true terminal observation before a reset.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from craftax.craftax_classic.constants import DIRECTIONS, Action
from craftax.craftax_classic.envs.common import compute_score
from craftax.craftax_classic.envs.craftax_pixels_env import (
    CraftaxClassicPixelsEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_state import EnvParams, EnvState
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
)
from craftax.craftax_classic.game_logic import craftax_step, get_player_attack_damage
from craftax.environment_base import spaces
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset


@struct.dataclass
class SeaEnvState:
    """JAX-compatible state added around the immutable Craftax state."""

    env_state: EnvState
    idle_steps: jax.Array


def _attacked_mob_mask(
    state: EnvState, positions: jax.Array, active: jax.Array, action: jax.Array
) -> jax.Array:
    target = state.player_position + DIRECTIONS[state.player_direction]
    is_in_target = jnp.all(positions == target, axis=-1)
    return is_in_target & active & (action == Action.DO.value)


def _restore_nonlethal_mob_damage(
    old_state: EnvState, new_state: EnvState, action: jax.Array
) -> EnvState:
    """Undo chip damage while retaining all side effects of a lethal hit.

    Craftax normally accumulates creature damage.  SEA's Crafter fork only
    subtracts health when ``attack_damage >= current_health``.  A failed hit
    therefore leaves health exactly unchanged.
    """

    damage = get_player_attack_damage(old_state)

    def restore(old_mobs, new_mobs):
        attacked = _attacked_mob_mask(
            old_state, old_mobs.position, old_mobs.mask, action
        )
        nonlethal = attacked & (damage < old_mobs.health)
        health = jnp.where(nonlethal, old_mobs.health, new_mobs.health)
        return new_mobs.replace(health=health)

    return new_state.replace(
        zombies=restore(old_state.zombies, new_state.zombies),
        cows=restore(old_state.cows, new_state.cows),
        skeletons=restore(old_state.skeletons, new_state.skeletons),
    )


class SeaCraftaxClassicEnvNoAutoReset(EnvironmentNoAutoReset):
    """Pixel or symbolic Craftax-Classic with SEA-compatible mechanics."""

    def __init__(
        self,
        symbolic: bool = True,
        idle_timeout: int = 100,
        static_env_params=None,
    ):
        super().__init__()
        self.symbolic = symbolic
        self.idle_timeout = int(idle_timeout)
        base_cls = (
            CraftaxClassicSymbolicEnvNoAutoReset
            if symbolic
            else CraftaxClassicPixelsEnvNoAutoReset
        )
        self._base_env = base_cls(static_env_params=static_env_params)
        self.static_env_params = self._base_env.static_env_params

    @property
    def default_params(self) -> EnvParams:
        # SEA's non-vanilla Crafter uses cow=2, skeleton=3 and zombie=5.
        return EnvParams(cow_health=2, skeleton_health=3, zombie_health=5)

    def step_env(
        self,
        key: jax.Array,
        state: SeaEnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[jax.Array, SeaEnvState, float, bool, dict]:
        old_state = state.env_state
        stepped_state, _ = craftax_step(
            key, old_state, action, params, self.static_env_params
        )

        stepped_state = _restore_nonlethal_mob_damage(old_state, stepped_state, action)
        # The original immortal setter immediately writes health=9.  Restore it
        # before rendering so neither symbolic nor pixel observations expose a
        # transient death state.
        stepped_state = stepped_state.replace(player_health=jnp.int32(9))

        new_achievements = stepped_state.achievements & ~old_state.achievements
        event_count = new_achievements.astype(jnp.float32).sum()
        event_happened = new_achievements.any()
        idle_steps = jnp.where(event_happened, 0, state.idle_steps + 1)

        idle_done = idle_steps >= self.idle_timeout
        time_done = stepped_state.timestep >= params.max_timesteps
        done = idle_done | time_done
        new_state = SeaEnvState(stepped_state, idle_steps)

        info = compute_score(stepped_state, done)
        info.update(
            {
                "discount": jnp.where(done, 0.0, 1.0),
                "event_count": event_count,
                "event_happened": event_happened,
                "new_achievements": new_achievements,
                "idle_steps": idle_steps,
                "idle_done": idle_done,
                "time_done": time_done,
            }
        )

        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            event_count,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, SeaEnvState]:
        _, core_state = self._base_env.reset_env(key, params)
        core_state = core_state.replace(player_health=jnp.int32(9))
        state = SeaEnvState(core_state, jnp.int32(0))
        return self.get_obs(state), state

    def get_obs(self, state: SeaEnvState) -> jax.Array:
        return self._base_env.get_obs(state.env_state)

    def is_terminal(self, state: SeaEnvState, params: EnvParams) -> bool:
        return (state.idle_steps >= self.idle_timeout) | (
            state.env_state.timestep >= params.max_timesteps
        )

    @property
    def name(self) -> str:
        mode = "Symbolic" if self.symbolic else "Pixels"
        return f"Craftax-Classic-SEA-{mode}-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return self._base_env.num_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return self._base_env.action_space(params)

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return self._base_env.observation_space(params or self.default_params)


def make_sea_craftax_classic_env(
    symbolic: bool = True, idle_timeout: int = 100
) -> SeaCraftaxClassicEnvNoAutoReset:
    return SeaCraftaxClassicEnvNoAutoReset(symbolic=symbolic, idle_timeout=idle_timeout)


__all__ = [
    "SeaCraftaxClassicEnvNoAutoReset",
    "SeaEnvState",
    "make_sea_craftax_classic_env",
]
