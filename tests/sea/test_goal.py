from pathlib import Path

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
)

ORACLE_PATH = Path(__file__).resolve().parents[2] / "craftax_classic_22.npz"


def _oracle_runtime(**kwargs):
    graph = jnp.zeros((22, 22), dtype=jnp.int8)
    options = dict(include_new_tasks=False)
    options.update(kwargs)
    return GoalRuntime(
        transition_network=None,
        transition_params=None,
        centroids=None,
        threshold=None,
        graph=graph,
        oracle_achievement_ids=tuple(range(22)),
        **options,
    )


def _step(runtime, objective, new, events=None, completed=None):
    new = jnp.asarray(new, dtype=bool)[None]
    events = new if events is None else jnp.asarray(events, dtype=bool)[None]
    width = runtime.num_objectives
    goal_state = GoalState(
        objective=jnp.asarray([objective], dtype=jnp.int32),
        completed=(
            jnp.zeros((1, width), dtype=bool)
            if completed is None
            else jnp.asarray(completed, dtype=bool)[None]
        ),
        last_completed=jnp.asarray([width], dtype=jnp.int32),
    )
    observation = jnp.zeros((1, 3))
    return apply_goal_transition(
        runtime,
        goal_state,
        jax.random.PRNGKey(0),
        observation,
        jnp.zeros((1,), dtype=jnp.int32),
        observation,
        jnp.zeros((1,), dtype=bool),
        new.astype(jnp.float32).sum(-1),
        new_achievements=new,
        achievement_events=events,
    )


def _flags(*ids):
    flags = np.zeros(22, dtype=bool)
    flags[list(ids)] = True
    return flags


def test_target_goal_reward_is_three_and_nontarget_is_one():
    runtime = _oracle_runtime()
    _, reward, goal_done, _, _ = _step(runtime, 0, _flags(0))
    assert goal_done[0] and reward[0] == 3.0
    _, reward, goal_done, _, _ = _step(runtime, 0, _flags(4))
    assert not goal_done[0] and reward[0] == 1.0
    _, reward, _, _, _ = _step(runtime, 0, _flags(0, 4, 9))
    assert reward[0] == 3.0 + 2.0
    _, reward, _, _, _ = _step(runtime, 0, _flags())
    assert reward[0] == 0.0


def test_repeated_target_counts_only_with_reselect():
    repeat = _flags(0)
    no_new = _flags()
    without = _oracle_runtime()
    _, reward, goal_done, _, _ = _step(without, 0, no_new, events=repeat)
    assert not goal_done[0] and reward[0] == 0.0
    with_reselect = _oracle_runtime(allow_reselect=True)
    _, reward, goal_done, _, _ = _step(with_reselect, 0, no_new, events=repeat)
    assert goal_done[0] and reward[0] == 3.0


def test_reselect_allows_completed_goals():
    graph = jnp.zeros((2, 2), dtype=jnp.int8)
    completed = jnp.asarray([[True, False]])
    last = jnp.asarray([2], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(1), 200)
    many = lambda reselect: jax.vmap(
        lambda k: select_goals(k[None], completed, last, graph, False, reselect)[0]
    )(keys)
    assert not (many(False) == 0).any()
    assert (many(True) == 0).any()


@pytest.mark.skipif(not ORACLE_PATH.exists(), reason="oracle graph file missing")
def test_load_oracle_graph_22():
    graph, ids = load_oracle_graph(ORACLE_PATH)
    assert graph.shape == (22, 22)
    assert ids == tuple(range(22))
    assert graph[3, 7] == 1  # collect_sapling -> place_plant


def test_oracle_unknown_goal_is_any_first_unlock():
    runtime = _oracle_runtime(include_new_tasks=True)
    unknown = runtime.num_clusters
    _, reward, goal_done, _, _ = _step(runtime, unknown, _flags(5, 9))
    assert goal_done[0] and reward[0] == 3.0 + 1.0
    _, reward, goal_done, _, _ = _step(
        runtime, unknown, _flags(), events=_flags(5)
    )
    assert not goal_done[0] and reward[0] == 0.0


def test_unknown_goal_prob_zero_never_selects_unknown():
    graph = jnp.zeros((2, 2), dtype=jnp.int8)
    completed = jnp.zeros((1, 3), dtype=bool)
    last = jnp.asarray([3], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(2), 300)
    pick = lambda prob: jax.vmap(
        lambda k: select_goals(k[None], completed, last, graph, True, False, prob)[0]
    )(keys)
    assert not (pick(0.0) == 2).any()
    assert (pick(0.4) == 2).any()
