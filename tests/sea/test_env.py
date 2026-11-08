import jax
import jax.numpy as jnp

from craftax.craftax_classic.constants import DIRECTIONS, Achievement, Action
from craftax.craftax_classic.constants import BlockType
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


def test_cow_health_stays_at_sea_value_in_cumulative_mode():
    params = make_sea_craftax_classic_env(
        combat_damage_mode="cumulative"
    ).default_params
    assert params.cow_health == 2


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


def test_combat_damage_mode_controls_nonlethal_accumulation(monkeypatch):
    from craftax.sea import env as sea_env

    def chip_damage(key, old, action, params, static_params):
        cows = old.cows.replace(health=old.cows.health.at[0].set(1))
        return old.replace(cows=cows, timestep=old.timestep + 1), 0.0

    monkeypatch.setattr(sea_env, "craftax_step", chip_damage)
    key = jax.random.PRNGKey(15)

    lethal_only = make_sea_craftax_classic_env(
        combat_damage_mode="lethal-only"
    )
    params = lethal_only.default_params
    _, state = lethal_only.reset(key, params)
    core = state.env_state
    state = state.replace(
        env_state=core.replace(cows=_put_in_front(core, core.cows, hp=2))
    )
    _, restored, _, _, _ = lethal_only.step_env(
        key, state, Action.DO.value, params
    )
    assert restored.env_state.cows.health[0] == 2

    cumulative = make_sea_craftax_classic_env(
        combat_damage_mode="cumulative"
    )
    _, accumulated, _, _, _ = cumulative.step_env(
        key, state, Action.DO.value, params
    )
    assert accumulated.env_state.cows.health[0] == 1


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


def test_mortal_health_reward_and_terminal_are_configurable(monkeypatch):
    from craftax.sea import env as sea_env

    env = make_sea_craftax_classic_env(
        immortal=False, health_reward_coef=0.5
    )
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(12), params)

    def lose_health(key, old, action, params, static_params):
        health = jnp.where(action == Action.NOOP.value, 7, 0)
        return old.replace(player_health=health, timestep=old.timestep + 1), 0.0

    monkeypatch.setattr(sea_env, "craftax_step", lose_health)
    _, next_state, reward, done, info = env.step(
        jax.random.PRNGKey(13), state, Action.NOOP.value, params
    )
    assert next_state.env_state.player_health == 7
    assert reward == -1.0
    assert not done
    assert not info["health_done"]

    _, _, reward, done, info = env.step(
        jax.random.PRNGKey(14), state, Action.DO.value, params
    )
    assert reward == -4.5
    assert done
    assert info["health_done"]


def test_immortal_environment_omits_health_shaping(monkeypatch):
    from craftax.sea import env as sea_env

    env = make_sea_craftax_classic_env(
        immortal=True, health_reward_coef=0.5
    )
    params = env.default_params
    _, state = env.reset(jax.random.PRNGKey(16), params)

    def lose_health(key, old, action, params, static_params):
        return old.replace(player_health=7, timestep=old.timestep + 1), 0.0

    monkeypatch.setattr(sea_env, "craftax_step", lose_health)
    _, next_state, reward, done, _ = env.step(
        jax.random.PRNGKey(17), state, Action.NOOP.value, params
    )
    assert next_state.env_state.player_health == 9
    assert reward == 0.0
    assert not done


def test_cli_exposes_immortal_and_health_reward_settings():
    from craftax.sea.train import build_parser

    args = build_parser().parse_args([
        "--immortal", "false", "--health-reward-coef", "0.1"
    ])
    assert args.immortal is False
    assert args.health_reward_coef == 0.1
    assert args.combat_damage_mode == "lethal-only"
    assert args.goal_success_coefficient == 5.0
    assert args.non_target_achievement_coefficient == 1.0

    from craftax.sea.train_goal_only import build_parser as build_goal_only_parser

    goal_only = build_goal_only_parser().parse_args([
        "source", "--output", "output"
    ])
    assert goal_only.goal_success_coefficient == 5.0
    assert goal_only.non_target_achievement_coefficient == 1.0

    cumulative = build_parser().parse_args([
        "--combat-damage-mode", "cumulative"
    ])
    assert cumulative.combat_damage_mode == "cumulative"


def test_repeated_drinking_is_an_event_without_repeating_episode_reward():
    env = make_sea_craftax_classic_env(
        symbolic=True, immortal=False, track_achievement_events=True,
    )
    _, state = env.reset(jax.random.PRNGKey(30))
    core = state.env_state
    target = core.player_position + DIRECTIONS[core.player_direction]
    core = core.replace(
        map=core.map.at[target[0], target[1]].set(BlockType.WATER.value),
        player_drink=jnp.int32(2),
    )
    state = state.replace(env_state=core)
    step = jax.jit(env.step_env)
    _, first, reward1, _, info1 = step(
        jax.random.PRNGKey(31), state, Action.DO.value, env.default_params,
    )
    _, second, reward2, _, info2 = step(
        jax.random.PRNGKey(32), first, Action.DO.value, env.default_params,
    )
    drink = Achievement.COLLECT_DRINK.value
    assert info1["new_achievements"][drink]
    assert info1["achievement_events"][drink]
    assert not info2["new_achievements"][drink]
    assert info2["achievement_events"][drink]
    assert second.env_state.achievements[drink]
    assert reward1 == 1.0 and reward2 == 0.0
    # Merely retaining the cumulative bit is not another performance event.
    _, _, _, _, idle_info = step(
        jax.random.PRNGKey(33), second, Action.NOOP.value, env.default_params,
    )
    assert not idle_info["achievement_events"][drink]


def test_event_tracking_preserves_physical_state_and_environment_reward():
    key = jax.random.PRNGKey(34)
    regular = make_sea_craftax_classic_env(symbolic=True)
    tracking = make_sea_craftax_classic_env(symbolic=True, track_achievement_events=True)
    _, state = regular.reset(key)
    core = state.env_state.replace(achievements=jnp.ones_like(state.env_state.achievements))
    state = state.replace(env_state=core)
    left = regular.step_env(key, state, Action.NOOP.value, regular.default_params)
    right = tracking.step_env(key, state, Action.NOOP.value, tracking.default_params)
    for x, y in zip(jax.tree.leaves(left[:4]), jax.tree.leaves(right[:4])):
        assert jnp.array_equal(x, y)
