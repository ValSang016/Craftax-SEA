"""Craftax-Classic with the environment semantics used by SEA.

This module reproduces the non-vanilla Crafter settings from the SEA codebase:

* the player is immortal and health is always nine;
* lava and health do not terminate the episode;
* an episode ends after 100 steps without a newly unlocked achievement;
* reward is the number of newly unlocked achievements (no health shaping);
* creature damage is configurable (SEA lethal-only by default, or cumulative); and
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

from craftax.craftax_classic.constants import DIRECTIONS, Achievement, Action
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


def _restore_immortal_player(old_state: EnvState, new_state: EnvState) -> EnvState:
    """Apply the modified Crafter immortal-health setter semantics.

    In the SEA fork a damage assignment leaves health at nine.  Consequently
    ``_wake_up_when_hurt`` also sees no health decrease.  Craftax damage code
    wakes the player directly, so that side effect must be undone as well.
    """

    hurt_wake = (
        old_state.is_sleeping
        & (old_state.player_energy < 9)
        & ~new_state.is_sleeping
    )
    wake_index = Achievement.WAKE_UP.value
    achievements = new_state.achievements.at[wake_index].set(
        jnp.where(
            hurt_wake,
            old_state.achievements[wake_index],
            new_state.achievements[wake_index],
        )
    )
    return new_state.replace(
        player_health=jnp.int32(9),
        is_sleeping=jnp.where(hurt_wake, True, new_state.is_sleeping),
        achievements=achievements,
    )


class SeaCraftaxClassicEnvNoAutoReset(EnvironmentNoAutoReset):
    """Pixel or symbolic Craftax-Classic with SEA-compatible mechanics."""

    def __init__(
        self,
        symbolic: bool = True,
        idle_timeout: int = 100,
        immortal: bool = True,
        health_reward_coef: float = 0.0,
        combat_damage_mode: str = "lethal-only",
        static_env_params=None,
        track_achievement_events: bool = False,
    ):
        super().__init__()
        self.symbolic = symbolic
        self.idle_timeout = int(idle_timeout)
        self.immortal = bool(immortal)
        self.health_reward_coef = float(health_reward_coef)
        self.track_achievement_events = bool(track_achievement_events)
        if combat_damage_mode not in {"lethal-only", "cumulative"}:
            raise ValueError(
                "combat_damage_mode must be 'lethal-only' or 'cumulative'"
            )
        self.combat_damage_mode = combat_damage_mode
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
        # Game logic uses achievement bits only for bookkeeping and reward.
        # Starting those bits empty exposes this step's actual events, including
        # repeats. Restore cumulative bits before observation/score/reward.
        step_input = (
            old_state.replace(achievements=jnp.zeros_like(old_state.achievements))
            if self.track_achievement_events else old_state
        )
        stepped_state, _ = craftax_step(
            key, step_input, action, params, self.static_env_params
        )

        if self.combat_damage_mode == "lethal-only":
            stepped_state = _restore_nonlethal_mob_damage(
                old_state, stepped_state, action
            )
        # The original immortal setter immediately writes health=9.  Restore it
        # before rendering so neither symbolic nor pixel observations expose a
        # transient death state.
        if self.immortal:
            stepped_state = _restore_immortal_player(step_input, stepped_state)

        achievement_events = stepped_state.achievements
        if self.track_achievement_events:
            stepped_state = stepped_state.replace(
                achievements=old_state.achievements | achievement_events
            )

        health_delta = (
            stepped_state.player_health - old_state.player_health
        ).astype(jnp.float32)
        new_achievements = stepped_state.achievements & ~old_state.achievements
        if not self.track_achievement_events:
            achievement_events = new_achievements
        event_count = new_achievements.astype(jnp.float32).sum()
        event_happened = new_achievements.any()
        idle_steps = jnp.where(event_happened, 0, state.idle_steps + 1)

        idle_done = idle_steps >= self.idle_timeout
        time_done = stepped_state.timestep >= params.max_timesteps
        health_done = (
            stepped_state.player_health <= 0
            if not self.immortal
            else jnp.asarray(False)
        ) # 체력 소진으로 끝났는지 확인하기 위한 variable
        done = idle_done | time_done | health_done
        reward = event_count + self.health_reward_coef * health_delta
        new_state = SeaEnvState(stepped_state, idle_steps)

        info = compute_score(stepped_state, done)
        info.update(
            {
                "discount": jnp.where(done, 0.0, 1.0),
                "event_count": event_count,
                "event_happened": event_happened,
                "new_achievements": new_achievements,
                "achievement_events": achievement_events,
                # The vector wrapper resets immediately after this call.  Keep
                # terminal episode state in info so PPO can aggregate the same
                # per-episode metrics as SEA's CrafterMonitorWrapper.
                "achievements": stepped_state.achievements,
                "episode_length": stepped_state.timestep,
                "idle_steps": idle_steps,
                "idle_done": idle_done,
                "time_done": time_done,
                "health_done": health_done,
            }
        )

        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
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
        return (
            (state.idle_steps >= self.idle_timeout)
            | (state.env_state.timestep >= params.max_timesteps)
            | (
                (state.env_state.player_health <= 0)
                if not self.immortal
                else jnp.asarray(False)
            )
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
    symbolic: bool = True,
    idle_timeout: int = 100,
    immortal: bool = True,
    health_reward_coef: float = 0.0,
    combat_damage_mode: str = "lethal-only",
    track_achievement_events: bool = False,
) -> SeaCraftaxClassicEnvNoAutoReset:
    return SeaCraftaxClassicEnvNoAutoReset(
        symbolic=symbolic,
        idle_timeout=idle_timeout,
        immortal=immortal,
        health_reward_coef=health_reward_coef,
        combat_damage_mode=combat_damage_mode,
        track_achievement_events=track_achievement_events,
    )


__all__ = [
    "SeaCraftaxClassicEnvNoAutoReset",
    "SeaEnvState",
    "_restore_immortal_player",
    "make_sea_craftax_classic_env",
]
