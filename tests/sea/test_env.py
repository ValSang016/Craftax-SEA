import jax
import jax.numpy as jnp

from craftax.craftax_classic.constants import DIRECTIONS, Achievement, Action
from craftax.sea.env import (
    _restore_immortal_player,
    _restore_nonlethal_mob_damage,
    make_sea_craftax_classic_env,
)


def _put_in_front(state, mobs, hp):
    target = state.player_position + DIRECTIONS[state.player_direction]
    return mobs.replace(
        position=mobs.position.at[0].set(target),
        health=mobs.health.at[0].set(hp),
        mask=mobs.mask.at[0].set(True),
    )


def test_default_params_match_sea_creature_health():
    params = make_sea_craftax_classic_env().default_params
    assert params.cow_health == 2
    assert params.skeleton_health == 3
    assert params.zombie_health == 5


def test_nonlethal_damage_is_not_accumulated():
    env = make_sea_craftax_classic_env()
    _, state = env.reset(jax.random.PRNGKey(0))
    old = state.env_state
    old = old.replace(zombies=_put_in_front(old, old.zombies, hp=5))
    damaged = old.replace(
        zombies=old.zombies.replace(health=old.zombies.health.at[0].set(4))
    )

    corrected = _restore_nonlethal_mob_damage(old, damaged, Action.DO.value)
    assert corrected.zombies.health[0] == 5


def test_equal_attack_damage_is_lethal():
    env = make_sea_craftax_classic_env()
    _, state = env.reset(jax.random.PRNGKey(1))
    old = state.env_state
    old = old.replace(cows=_put_in_front(old, old.cows, hp=2))
    old = old.replace(inventory=old.inventory.replace(wood_sword=1))
    killed = old.replace(
        cows=old.cows.replace(
            health=old.cows.health.at[0].set(0),
            mask=old.cows.mask.at[0].set(False),
        )
    )

    corrected = _restore_nonlethal_mob_damage(old, killed, Action.DO.value)
    assert corrected.cows.health[0] == 0
    assert not corrected.cows.mask[0]


def test_immortality_does_not_create_hurt_wake_up_event():
    env = make_sea_craftax_classic_env()
    _, state = env.reset(jax.random.PRNGKey(11))
    old = state.env_state.replace(is_sleeping=jnp.array(True), player_energy=1)
    wake_index = Achievement.WAKE_UP.value
    attacked = old.replace(
        player_health=2,
        is_sleeping=jnp.array(False),
        achievements=old.achievements.at[wake_index].set(True),
    )
    corrected = _restore_immortal_player(old, attacked)
    assert corrected.player_health == 9
    assert corrected.is_sleeping
    assert not corrected.achievements[wake_index]


def test_reset_and_step_keep_player_immortal():
    env = make_sea_craftax_classic_env()
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(2), params)
    state = state.replace(env_state=state.env_state.replace(player_health=0))
    _, next_state, reward, done, info = env.step(
        jax.random.PRNGKey(3), state, Action.NOOP.value, params
    )
    assert next_state.env_state.player_health == 9
    assert reward == 0
    assert not done
    assert info["achievements"].shape == (22,)
    assert info["episode_length"] == next_state.env_state.timestep


def test_idle_timeout_is_exactly_one_hundred_steps():
    env = make_sea_craftax_classic_env(idle_timeout=100)
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(4), params)
    state = state.replace(idle_steps=jnp.int32(99))
    _, next_state, _, done, info = env.step(
        jax.random.PRNGKey(5), state, Action.NOOP.value, params
    )
    assert next_state.idle_steps == 100
    assert done
    assert info["idle_done"]
