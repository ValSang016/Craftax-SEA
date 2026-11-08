import jax
import jax.numpy as jnp
import numpy as np
import pytest

from craftax.sea.goal import (
    GoalRuntime,
    GoalState,
    apply_goal_transition,
    load_oracle_graph,
    select_goals,
    shape_goal_reward,
    advance_goal_progress,
)


def test_default_goal_reward_scales_target_and_rewards_only_non_target_events():
    runtime = _runtime()
    target = shape_goal_reward(runtime, jnp.asarray([True]), jnp.asarray([1.0]))
    non_target = shape_goal_reward(
        runtime, jnp.asarray([False]), jnp.asarray([1.0])
    )
    target_with_other = shape_goal_reward(
        runtime, jnp.asarray([True]), jnp.asarray([2.0])
    )
    healing = shape_goal_reward(
        runtime,
        jnp.asarray([False]),
        jnp.asarray([0.0]),
        health_reward=jnp.asarray([0.2]),
    )
    damage = shape_goal_reward(
        runtime,
        jnp.asarray([False]),
        jnp.asarray([0.0]),
        health_reward=jnp.asarray([-0.2]),
    )
    np.testing.assert_allclose(target, [5.0])
    np.testing.assert_allclose(non_target, [1.0])
    np.testing.assert_allclose(target_with_other, [6.0])
    np.testing.assert_allclose(healing, [0.2])
    np.testing.assert_allclose(damage, [-0.2])


def _runtime():
    return GoalRuntime(
        transition_network=None, transition_params=None,
        centroids=jnp.zeros((2, 1)), threshold=1.0,
        graph=jnp.zeros((2, 2), dtype=jnp.int8), include_new_tasks=True,
        oracle_achievement_ids=(0, 1),
    )


@pytest.mark.parametrize("allow_completed", [False, True])
def test_repeated_target_success_does_not_erase_simultaneous_novel_reward(allow_completed):
    runtime = GoalRuntime(
        transition_network=None, transition_params=None,
        centroids=jnp.zeros((2, 1)), threshold=0.0,
        graph=jnp.asarray([[0, 1], [0, 0]]), include_new_tasks=False,
        oracle_achievement_ids=(0, 1), allow_completed_goals=allow_completed,
    )
    goal = GoalState(
        objective=jnp.asarray([0]), completed=jnp.asarray([[True, False]]),
        last_completed=jnp.asarray([1]),
    )
    novel = jnp.zeros((1, 22), bool).at[0, 1].set(True)
    events = novel.at[0, 0].set(True)
    result = advance_goal_progress(
        runtime, goal, jnp.zeros((1, 1)), jnp.asarray([0]),
        jnp.zeros((1, 1)), jnp.asarray([False]), jnp.asarray([1.0]),
        new_achievements=novel, achievement_events=events,
    )
    state, reward, success = result[:3]
    assert bool(success[0]) == allow_completed
    assert reward[0] == 1.0
    np.testing.assert_array_equal(state.completed, [[True, True]])
    assert state.last_completed[0] == (0 if allow_completed else 1)


@pytest.mark.parametrize("new_target", [False, True])
def test_target_bonus_is_first_achievement_only_even_when_success_repeats(new_target):
    reward = shape_goal_reward(
        _runtime(), goal_done=jnp.asarray([True]),
        achievement_reward=jnp.asarray([float(new_target)]),
        target_is_new=jnp.asarray([new_target]), health_reward=jnp.asarray([0.2]),
    )
    np.testing.assert_allclose(reward, [5.2 if new_target else 0.2])


def test_repeated_oracle_target_ends_option_with_zero_achievement_reward():
    runtime = GoalRuntime(
        transition_network=None, transition_params=None,
        centroids=jnp.zeros((2, 1)), threshold=0.0,
        graph=jnp.zeros((2, 2)), include_new_tasks=False,
        oracle_achievement_ids=(0, 1), allow_completed_goals=True,
    )
    goal = GoalState(jnp.asarray([0]), jnp.asarray([[True, False]]), jnp.asarray([1]))
    result = advance_goal_progress(
        runtime, goal, jnp.zeros((1, 1)), jnp.asarray([0]), jnp.zeros((1, 1)),
        jnp.asarray([False]), jnp.asarray([0.0]),
        new_achievements=jnp.zeros((1, 22), bool),
        achievement_events=jnp.zeros((1, 22), bool).at[0, 0].set(True),
    )
    assert result[2][0]
    assert result[1][0] == 0.0


def test_load_oracle_graph_validates_mapping(tmp_path):
    path = tmp_path / "oracle.npz"
    np.savez(
        path,
        graph=np.asarray([[0, 1], [0, 0]], dtype=np.int8),
        achievement_ids=np.asarray([0, 5], dtype=np.int32),
        achievement_names=np.asarray(["collect_wood", "make_wood_pickaxe"]),
    )
    oracle = load_oracle_graph(path)
    assert oracle.graph.tolist() == [[0, 1], [0, 0]]
    assert oracle.achievement_ids == (0, 5)
    assert oracle.achievement_names == ("collect_wood", "make_wood_pickaxe")


def test_load_oracle_graph_rejects_cycle(tmp_path):
    path = tmp_path / "cycle.npz"
    np.savez(
        path,
        graph=np.asarray([[0, 1], [1, 0]], dtype=np.int8),
        achievement_ids=np.asarray([0, 1], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="DAG"):
        load_oracle_graph(path)


def test_oracle_transition_without_unknown_objective():
    runtime = GoalRuntime(
        transition_network=None,
        transition_params=None,
        centroids=jnp.zeros((2, 1)),
        threshold=0.0,
        graph=jnp.asarray([[0, 1], [0, 0]], dtype=jnp.int8),
        include_new_tasks=False,
        oracle_achievement_ids=(0, 5),
    )
    state = GoalState(
        objective=jnp.asarray([0]),
        completed=jnp.zeros((1, 2), dtype=bool),
        last_completed=jnp.asarray([2]),
    )
    achievements = jnp.zeros((1, 22), dtype=bool).at[0, 0].set(True)
    next_state, _, done, predicted, _ = apply_goal_transition(
        runtime,
        state,
        jax.random.PRNGKey(7),
        jnp.zeros((1, 1)),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, 1)),
        jnp.zeros((1,), dtype=bool),
        jnp.ones((1,)),
        new_achievements=achievements,
    )
    assert bool(done[0])
    assert int(predicted[0]) == 0
    assert next_state.completed.shape == (1, 2)
