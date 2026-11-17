"""Craftax-Classic with the environment semantics used by SEA.

This module reproduces the non-vanilla Crafter settings from the SEA codebase:

* the player is immortal and health is always nine (``mortal=False``,
  default); with ``mortal=True`` the vanilla Craftax health dynamics apply
  and death (health <= 0) or standing in lava terminates the episode;
* otherwise lava and health do not terminate the episode;
* an episode ends after 100 steps without a newly unlocked achievement;
* reward is the number of newly unlocked achievements (no health shaping);
* creatures only take damage when the current hit is lethal and cow,
  skeleton and zombie health are 2, 3 and 5 (``combat="sea"``, default); with
  ``combat="org"`` damage accumulates as in vanilla Craftax and the vanilla
  creature health (cow 3, skeleton 3, zombie 5) is used.

``info["achievement_events"]`` marks every achievement event of a step,
including repeats of an already unlocked achievement, whereas
``info["new_achievements"]`` only marks first unlocks within the episode.

The environment never auto-resets.  This is important for SEA because the
transition encoder must see the true terminal observation before a reset.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from craftax.craftax_classic.constants import (
    DIRECTIONS,
    Achievement,
    Action,
    BlockType,
)
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


COMBAT_MODES = ("sea", "org")
IMMORTAL_IDLE_TIMEOUT = 100
MORTAL_IDLE_TIMEOUT = 10_000
# Crafter rewards 0.1 per health point next to +1 per achievement; the goal
# reward here is +3 per target, so mortal runs use 0.3 to keep that ratio.
MORTAL_HEALTH_REWARD_COEF = 0.3


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
        idle_timeout: Optional[int] = None,
        static_env_params=None,
        mortal: bool = False,
        combat: str = "sea",
        health_reward_coef: Optional[float] = None,
    ):
        super().__init__()
        if combat not in COMBAT_MODES:
            raise ValueError(f"combat must be one of {COMBAT_MODES}, got {combat!r}")
        self.combat = combat
        self.symbolic = symbolic
        self.mortal = bool(mortal)
        if idle_timeout is None:
            idle_timeout = default_idle_timeout(self.mortal)
        if int(idle_timeout) < 1:
            raise ValueError("idle_timeout must be positive")
        self.idle_timeout = int(idle_timeout)
        if health_reward_coef is None:
            health_reward_coef = default_health_reward_coef(self.mortal)
        health_reward_coef = float(health_reward_coef)
        if not math.isfinite(health_reward_coef) or health_reward_coef < 0:
            raise ValueError("health_reward_coef must be finite and non-negative")
        self.health_reward_coef = health_reward_coef
        base_cls = (
            CraftaxClassicSymbolicEnvNoAutoReset
            if symbolic
            else CraftaxClassicPixelsEnvNoAutoReset
        )
        self._base_env = base_cls(static_env_params=static_env_params)
        self.static_env_params = self._base_env.static_env_params

    @property
    def default_params(self) -> EnvParams:
        if self.combat == "org":
            # Vanilla Craftax-Classic creature health (cow=3, skeleton=3, zombie=5).
            return EnvParams()
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
        # Craftax only ORs achievement flags and never reads them for game
        # logic.  Stepping with cleared flags therefore yields exactly the
        # achievement events of this step, including repeated events.
        cleared = old_state.replace(
            achievements=jnp.zeros_like(old_state.achievements)
        )
        stepped_state, _ = craftax_step(
            key, cleared, action, params, self.static_env_params
        )
        achievement_events = stepped_state.achievements
        stepped_state = stepped_state.replace(
            achievements=old_state.achievements | achievement_events
        )

        if self.combat == "sea":
            stepped_state = _restore_nonlethal_mob_damage(
                old_state, stepped_state, action
            )
        # The original immortal setter immediately writes health=9.  Restore it
        # before rendering so neither symbolic nor pixel observations expose a
        # transient death state.  Mortal mode keeps vanilla Craftax health.
        if not self.mortal:
            stepped_state = _restore_immortal_player(old_state, stepped_state)
            # A hurt wake-up that the immortal setter undid is not an event.
            wake_index = Achievement.WAKE_UP.value
            achievement_events = achievement_events.at[wake_index].set(
                achievement_events[wake_index] & ~stepped_state.is_sleeping
            )

        new_achievements = stepped_state.achievements & ~old_state.achievements
        event_count = new_achievements.astype(jnp.float32).sum()
        event_happened = new_achievements.any()
        idle_steps = jnp.where(event_happened, 0, state.idle_steps + 1)

        idle_done = idle_steps >= self.idle_timeout
        time_done = stepped_state.timestep >= params.max_timesteps
        death_done = self._death_done(stepped_state)
        done = idle_done | time_done | death_done
        new_state = SeaEnvState(stepped_state, idle_steps)

        health_delta = stepped_state.player_health.astype(jnp.float32) - (
            old_state.player_health.astype(jnp.float32)
        )

        info = compute_score(stepped_state, done)
        info.update(
            {
                # Crafter-style health shaping, added to the worker reward by
                # the trainers (the returned env reward stays the event count).
                "health_reward": self.health_reward_coef * health_delta,
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
                "death_done": death_done,
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

    def _death_done(self, state: EnvState) -> jax.Array:
        """Vanilla Craftax death: health <= 0 or standing in lava."""
        if not self.mortal:
            return jnp.array(False)
        in_lava = (
            state.map[state.player_position[0], state.player_position[1]]
            == BlockType.LAVA.value
        )
        return (state.player_health <= 0) | in_lava

    def is_terminal(self, state: SeaEnvState, params: EnvParams) -> bool:
        return (
            (state.idle_steps >= self.idle_timeout)
            | (state.env_state.timestep >= params.max_timesteps)
            | self._death_done(state.env_state)
        )

    @property
    def name(self) -> str:
        mode = "Symbolic" if self.symbolic else "Pixels"
        mortality = "-Mortal" if self.mortal else ""
        combat = "-CombatOrg" if self.combat == "org" else ""
        return f"Craftax-Classic-SEA{mortality}{combat}-{mode}-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return self._base_env.num_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return self._base_env.action_space(params)

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return self._base_env.observation_space(params or self.default_params)


def default_idle_timeout(mortal: bool) -> int:
    """Idle timeout used when none is given: 100 immortal, 10000 mortal."""
    return MORTAL_IDLE_TIMEOUT if mortal else IMMORTAL_IDLE_TIMEOUT


def default_health_reward_coef(mortal: bool) -> float:
    """Health shaping used when none is given: 0.3 mortal, 0 immortal."""
    return MORTAL_HEALTH_REWARD_COEF if mortal else 0.0


def make_sea_craftax_classic_env(
    symbolic: bool = True,
    idle_timeout: Optional[int] = None,
    mortal: bool = False,
    combat: str = "sea",
    health_reward_coef: Optional[float] = None,
) -> SeaCraftaxClassicEnvNoAutoReset:
    if idle_timeout is None:
        idle_timeout = default_idle_timeout(mortal)
    return SeaCraftaxClassicEnvNoAutoReset(
        symbolic=symbolic, idle_timeout=idle_timeout, mortal=mortal, combat=combat,
        health_reward_coef=health_reward_coef,
    )


__all__ = [
    "COMBAT_MODES",
    "SeaCraftaxClassicEnvNoAutoReset",
    "default_health_reward_coef",
    "default_idle_timeout",
    "SeaEnvState",
    "_restore_immortal_player",
    "make_sea_craftax_classic_env",
]
