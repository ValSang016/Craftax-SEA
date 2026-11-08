import jax
import jax.numpy as jnp
import numpy as np
import pytest

from craftax.sea.dynamic_graph import (
    HighLevelReplayBuffer,
    HighLevelTransitionBatch,
    base_q_regression_loss,
    compute_dynamic_edge_scores,
    compute_q_update_count,
    compute_smdp_target,
    get_ready_goal_candidates,
    get_goal_candidates,
    high_level_option_done,
    initialize_high_level_q,
    split_symbolic_state,
    stable_masked_softmax,
    update_high_level_q,
)


def test_split_symbolic_state_uses_classic_intrinsic_slice():
    state = jnp.arange(1345, dtype=jnp.float32)[None]
    external, internal = split_symbolic_state(state)
    assert external.shape == (1, 1341)
    np.testing.assert_array_equal(internal, state[:, 1335:1339])
    np.testing.assert_array_equal(external[:, 1334:1337], state[:, [1334, 1339, 1340]])


def test_internal_slice_matches_classic_renderer_intrinsics():
    from craftax.sea.env import make_sea_craftax_classic_env

    env = make_sea_craftax_classic_env(symbolic=True)
    observation, state = env.reset(jax.random.PRNGKey(11))
    core = state.env_state
    expected = jnp.asarray(
        [
            core.player_health,
            core.player_food,
            core.player_drink,
            core.player_energy,
        ],
        dtype=jnp.float16,
    ) / 10.0
    assert observation.shape == (1345,)
    np.testing.assert_array_equal(observation[1335:1339], expected)


def test_candidates_include_only_unfinished_ready_nodes_and_unknown():
    graph = jnp.asarray(
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.int8
    )
    initial = get_ready_goal_candidates(
        jnp.asarray([[False, False, False, False]]), graph, True
    )
    np.testing.assert_array_equal(initial, [[True, False, True, True]])

    after_root = get_ready_goal_candidates(
        jnp.asarray([[True, False, False, False]]), graph, True
    )
    np.testing.assert_array_equal(after_root, [[False, True, True, True]])


@pytest.mark.parametrize(
    "allow_unready,allow_completed,expected",
    [
        (False, False, [False, True, False]),
        (False, True, [True, True, False]),
        (True, False, [False, True, True]),
        (True, True, [True, True, True]),
    ],
)
def test_goal_eligibility_options_are_independent(allow_unready, allow_completed, expected):
    graph = jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=jnp.int8)
    mask = get_goal_candidates(
        jnp.asarray([[True, False, False]]), graph, False,
        allow_unready_goals=allow_unready,
        allow_completed_goals=allow_completed,
    )
    np.testing.assert_array_equal(mask, [expected])


@pytest.mark.parametrize("allow_completed", [False, True])
def test_warmup_excludes_unready_even_when_allowed_afterwards(allow_completed):
    graph = jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=jnp.int8)
    completed = jnp.asarray([[True, False, False]])
    evaluate = jax.jit(lambda warmup: get_goal_candidates(
        completed, graph, False, allow_unready_goals=True,
        allow_completed_goals=allow_completed, warmup=warmup,
    ))
    np.testing.assert_array_equal(evaluate(True), [[allow_completed, True, False]])
    np.testing.assert_array_equal(evaluate(False), [[allow_completed, True, True]])


def test_dynamic_score_is_static_edge_plus_weighted_residual():
    graph = jnp.asarray([[0, 1], [0, 0]], dtype=jnp.int8)
    full = jnp.asarray([[3.0, 4.0, 2.0]])
    base = jnp.asarray([[1.0, 3.0, 3.0]])
    scores = compute_dynamic_edge_scores(
        graph, jnp.asarray([0]), full, base, residual_weight=0.5
    )
    np.testing.assert_allclose(scores, [[1.0, 1.5, -0.5]])


def test_masked_softmax_is_uniform_or_candidate_masked():
    candidates = jnp.asarray([[True, False, True]])
    uniform = stable_masked_softmax(jnp.zeros((1, 3)), candidates)
    dynamic = stable_masked_softmax(
        jnp.asarray([[1000.0, 5000.0, 999.0]]), candidates
    )
    np.testing.assert_allclose(uniform, [[0.5, 0.0, 0.5]])
    assert dynamic[0, 1] == 0
    assert jnp.isfinite(dynamic).all()
    np.testing.assert_allclose(dynamic.sum(-1), [1.0])


def test_smdp_target_uses_duration_terminal_and_candidate_mask():
    target = compute_smdp_target(
        reward=jnp.asarray([1.0, 1.0]),
        duration=jnp.asarray([2, 2]),
        done=jnp.asarray([False, True]),
        next_q=jnp.asarray([[2.0, 100.0], [2.0, 100.0]]),
        candidates=jnp.asarray([[True, False], [True, False]]),
        gamma=0.5,
    )
    np.testing.assert_allclose(target, [1.5, 1.0])


def test_option_ends_only_on_goal_success_timeout_or_episode_end():
    ended = high_level_option_done(
        goal_done=jnp.asarray([False, False, True, False]),
        duration=jnp.asarray([299, 300, 1, 1]),
        env_done=jnp.asarray([False, False, False, True]),
        goal_timeout=300,
    )
    np.testing.assert_array_equal(ended, [False, True, True, True])


def test_q_update_count_scales_with_new_transitions_and_replay_ratio():
    assert compute_q_update_count(0, 256, 4.0) == 0
    assert compute_q_update_count(1, 256, 4.0) == 1
    assert compute_q_update_count(128, 256, 4.0) == 2
    assert compute_q_update_count(257, 256, 4.0) == 5


def test_base_q_target_stops_gradient_to_full_q():
    base = jnp.asarray([1.0, 2.0])
    full = jnp.asarray([3.0, 4.0])
    full_gradient = jax.grad(base_q_regression_loss, argnums=1)(base, full)
    np.testing.assert_array_equal(full_gradient, jnp.zeros_like(full))


@pytest.mark.parametrize(
    "allow_unready,allow_completed,warmup,expected_loss",
    [(False, False, False, 1.0), (True, False, False, 400.0),
     (True, False, True, 1.0), (False, True, False, 10000.0)],
)
def test_q_bootstrap_uses_the_configured_goal_eligibility(
    allow_unready, allow_completed, warmup, expected_loss,
):
    obs = jnp.zeros((1, 1345))
    completed = jnp.asarray([[True, False, False]])
    node = jnp.asarray([0])
    q = initialize_high_level_q(jax.random.PRNGKey(44), obs, completed, node, 3, 8, 1e-3)
    zeros = jax.tree.map(jnp.zeros_like, q.full_q.params)
    target = jax.tree.map(jnp.zeros_like, q.target_full_params)
    target["params"]["Dense_2"]["bias"] = jnp.asarray([100.0, 1.0, 20.0])
    q = q.replace(full_q=q.full_q.replace(params=zeros), target_full_params=target)
    batch = HighLevelTransitionBatch(
        obs, completed, node, jnp.asarray([1]), jnp.zeros(1), obs, completed,
        node, jnp.ones(1, jnp.int32), jnp.zeros(1, bool), jnp.zeros(1, bool),
    )
    graph = jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    _, metrics = update_high_level_q(
        q, batch, graph, False, 1.0,
        allow_unready_goals=allow_unready, allow_completed_goals=allow_completed,
        warmup=warmup,
    )
    assert metrics["full_q_loss"] == expected_loss


def test_full_q_has_internal_state_inputs_and_base_q_does_not():
    observation = jnp.zeros((2, 1345), dtype=jnp.float32)
    completed = jnp.zeros((2, 3), dtype=bool)
    current = jnp.full((2,), 3, dtype=jnp.int32)
    state = initialize_high_level_q(
        jax.random.PRNGKey(4),
        observation,
        completed,
        current,
        num_objectives=3,
        hidden_size=8,
        learning_rate=1e-3,
    )
    full_width = state.full_q.params["params"]["Dense_0"]["kernel"].shape[0]
    base_width = state.base_q.params["params"]["Dense_0"]["kernel"].shape[0]
    assert full_width == base_width + 4


def test_replay_stores_raw_state_once_and_samples_transition_fields():
    buffer = HighLevelReplayBuffer.create(4, (1345,), 3)
    observation = jnp.arange(2 * 1345, dtype=jnp.float32).reshape(2, 1345)
    transition = HighLevelTransitionBatch(
        observation=observation,
        completed=jnp.zeros((2, 3), dtype=bool),
        current_node=jnp.asarray([3, 0]),
        goal=jnp.asarray([0, 1]),
        reward=jnp.asarray([1.0, 2.0]),
        next_observation=observation + 1,
        next_completed=jnp.zeros((2, 3), dtype=bool),
        next_current_node=jnp.asarray([0, 1]),
        duration=jnp.asarray([2, 3]),
        done=jnp.asarray([False, True]),
        success=jnp.asarray([True, False]),
    )
    buffer = buffer.add(transition, jnp.asarray([True, False]))
    assert int(buffer.size) == 1
    assert buffer.observation.shape == (4, 1345)
    assert not hasattr(buffer, "external_state")
    assert not hasattr(buffer, "internal_state")
    sampled = buffer.sample(jax.random.PRNGKey(0), 2)
    np.testing.assert_array_equal(sampled.goal, [0, 0])
    np.testing.assert_array_equal(sampled.duration, [2, 2])


def test_vectorized_replay_append_keeps_latest_values_across_wraparound():
    count = 6
    values = jnp.arange(count, dtype=jnp.int32)
    observation = values[:, None].astype(jnp.float32)
    transition = HighLevelTransitionBatch(
        observation=observation,
        completed=jnp.zeros((count, 1), dtype=bool),
        current_node=values,
        goal=values,
        reward=values.astype(jnp.float32),
        next_observation=observation + 1,
        next_completed=jnp.zeros((count, 1), dtype=bool),
        next_current_node=values + 1,
        duration=values + 1,
        done=jnp.zeros((count,), dtype=bool),
        success=jnp.ones((count,), dtype=bool),
    )
    buffer = HighLevelReplayBuffer.create(4, (1,), 1)
    buffer = jax.jit(lambda value: value.add(transition, values >= 0))(buffer)
    assert int(buffer.size) == 4
    assert int(buffer.position) == 2
