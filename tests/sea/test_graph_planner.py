from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from craftax.sea.dynamic_graph import DynamicGraphConfig, HighLevelTransitionBatch
from craftax.sea.graph_planner import (
    GraphPlannerState,
    OptionModelOutput,
    choose_planner_goals,
    initialize_graph_planner,
    plan_path_scores,
    update_graph_planner,
)
from craftax.sea.train import _make_dynamic_config, build_parser


def _toy_state():
    """Goal 1 is only feasible after goal 0; goal 2 is a moderately easy leaf.

    The latent is the completed vector, so the toy model is exact.
    """

    def apply_fn(params, *inputs, method=None):
        name = method.__name__
        if name == "encode":
            return inputs[1].astype(jnp.float32)
        if name == "transition":
            latent, goal = inputs
            p = jnp.where(
                goal == 0, 0.6,
                jnp.where(goal == 2, 0.7, jnp.where(latent[..., 0] > 0, 0.99, 0.01)),
            )
            next_latent = jnp.maximum(latent, jax.nn.one_hot(goal, latent.shape[-1]))
            return OptionModelOutput(
                success_logit=jnp.log(p) - jnp.log1p(-p),
                log_duration=jnp.zeros_like(p),
                death_logit=jnp.full_like(p, -10.0),
                completed_logits=jnp.full(p.shape + (3,), -10.0),
                next_latent=next_latent,
            )
        raise AssertionError(name)

    train_state = TrainState.create(apply_fn=apply_fn, params={}, tx=optax.sgd(0.0))
    return GraphPlannerState(train_state, {})


def _scores(horizon, use_leaf=False, completed=None):
    completed = jnp.zeros((1, 3), dtype=bool) if completed is None else completed
    return plan_path_scores(
        _toy_state(),
        jnp.zeros((1, 4)),
        completed,
        jnp.array([3]),
        jnp.ones((1, 3), dtype=bool),
        horizon=horizon,
        beam_width=2,
        gamma=1.0,
        use_leaf=use_leaf,
        max_duration=300.0,
    )[0]


def test_one_hop_is_greedy_but_two_hops_value_unlocking():
    np.testing.assert_allclose(_scores(1), [0.6, 0.01, 0.7], atol=1e-5)
    two_hop = _scores(2)
    # 0 -> 1 : 0.6 + 0.6 * 0.99 ; 2 -> 0 : 0.7 + 0.7 * 0.6
    np.testing.assert_allclose(two_hop[0], 0.6 + 0.6 * 0.99, atol=1e-5)
    np.testing.assert_allclose(two_hop[2], 0.7 + 0.7 * 0.6, atol=1e-5)
    assert int(jnp.argmax(two_hop)) == 0


def test_completed_goals_earn_no_score_and_leaf_adds_out_degree():
    completed = jnp.array([[False, False, True]])
    assert float(_scores(1, completed=completed)[2]) == 0.0
    leaf = _scores(1, use_leaf=True)
    # After goal 0 succeeds, goals 1 and 2 are both new: 0.99 + 0.7.
    np.testing.assert_allclose(leaf[0], 0.6 + 0.6 * (0.99 + 0.7), atol=1e-5)


def test_planner_budget_overflow_uses_one_hop_fallback():
    config = DynamicGraphConfig(
        controller="graph-planner", planner_horizon=3, planner_max_decisions=2
    )
    needs_goal = jnp.array([True, True, True, False])
    selected, overflow = choose_planner_goals(
        jax.random.PRNGKey(0), _toy_state(), jnp.zeros((4, 4)),
        jnp.zeros((4, 3), dtype=bool), jnp.full((4,), 3),
        jnp.ones((4, 3), dtype=bool), needs_goal, config,
    )
    assert selected.shape == (4,)
    assert int(overflow.sum()) == 1
    assert not bool(overflow[3])


def test_supervised_update_is_finite_and_moves_target():
    config = DynamicGraphConfig(
        controller="graph-planner", model_hidden_size=16, model_latent_size=8
    )
    batch_size, goals = 6, 4
    observation = jax.random.uniform(jax.random.PRNGKey(1), (batch_size, 1345))
    completed = jnp.zeros((batch_size, goals), dtype=bool)
    node = jnp.full((batch_size,), goals)
    state = initialize_graph_planner(
        jax.random.PRNGKey(2), observation, completed, node, goals, config
    )
    batch = HighLevelTransitionBatch(
        observation=observation,
        completed=completed,
        current_node=node,
        goal=jnp.arange(batch_size) % goals,
        reward=jnp.zeros((batch_size,)),
        next_observation=observation[::-1],
        next_completed=completed.at[:, 0].set(True),
        next_current_node=jnp.zeros((batch_size,), dtype=jnp.int32),
        duration=jnp.arange(1, batch_size + 1),
        done=jnp.zeros((batch_size,), dtype=bool),
        success=jnp.arange(batch_size) % 2 == 0,
        died=jnp.zeros((batch_size,), dtype=bool),
    )
    new_state, metrics = update_graph_planner(state, batch, 0.5, goals)
    assert all(np.isfinite(float(value)) for value in metrics.values())
    moved = jax.tree.map(
        lambda a, b: float(jnp.abs(a - b).max()),
        new_state.target_params, state.target_params,
    )
    assert max(jax.tree.leaves(moved)) > 0


def test_cli_builds_graph_planner_config():
    args = build_parser().parse_args([
        "--symbolic", "--high-level-controller", "graph-planner",
        "--planner-horizon", "4", "--planner-leaf", "false",
        "--allow-completed-goals", "true",
    ])
    config = _make_dynamic_config(args)
    assert config.controller == "graph-planner"
    assert config.planner_horizon == 4 and not config.planner_leaf
    assert replace(config, planner_horizon=2).planner_horizon == 2


def _oracle_runtime(repeat=0.0):
    from craftax.sea.goal import GoalRuntime

    return GoalRuntime(
        transition_network=None, transition_params=None,
        centroids=jnp.zeros((3, 1)), threshold=0.0,
        graph=jnp.zeros((3, 3), dtype=jnp.int8),
        include_new_tasks=False, oracle_achievement_ids=(0, 1, 2),
        allow_completed_goals=True, repeat_target_coefficient=repeat,
    )


def _progress(runtime, objective, last, new, events):
    from craftax.sea.goal import GoalState, advance_goal_progress

    state = GoalState(
        jnp.array([objective]), jnp.array([[True, True, False]]), jnp.array([last])
    )
    padded = lambda x: jnp.zeros((1, 22), dtype=bool).at[0, :3].set(jnp.array(x))
    return advance_goal_progress(
        runtime, state, None, None, None, jnp.array([False]),
        jnp.array([float(sum(new))]),
        new_achievements=padded(new), achievement_events=padded(events),
    )


def test_incidental_repeat_does_not_move_current_node():
    result = _progress(_oracle_runtime(), objective=2, last=1,
                       new=[False] * 3, events=[True, False, False])
    assert int(result[0].last_completed[0]) == 1


def test_commanded_repeat_moves_node_and_gets_repeat_reward():
    result = _progress(_oracle_runtime(repeat=1.0), objective=0, last=1,
                       new=[False] * 3, events=[True, False, False])
    assert int(result[0].last_completed[0]) == 0
    assert float(result[1][0]) == 1.0
    first_time = _progress(_oracle_runtime(repeat=1.0), objective=2, last=1,
                           new=[False, False, True], events=[False, False, True])
    assert float(first_time[1][0]) == 5.0


def test_softmax_is_scale_free():
    from craftax.sea.graph_planner import sample_from_scores

    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    candidates = jnp.ones((22,), dtype=bool)
    for scale in (0.03, 3.0):
        scores = jnp.zeros((22,)).at[5].set(scale).at[6].set(0.5 * scale)
        picks = jax.vmap(lambda k: sample_from_scores(k, scores, candidates, 0.1))(keys)
        assert float((picks == 5).mean()) > 0.97
