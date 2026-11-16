from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from craftax.sea.goal import GoalState, load_oracle_graph
from craftax.sea.ppo_rnn import PPOConfig
from craftax.sea_planner.config import PlannerConfig
from craftax.sea_planner.goals import (
    PlannerGoalRuntime,
    advance_goal_progress,
    get_goal_candidates,
    high_level_option_done,
    warmup_option_done,
    warmup_option_recorded,
)
from craftax.sea_planner.metrics import NO_EAT_PLANT_INDICES, scores_from_counts
from craftax.sea_planner.option_model import (
    EdgePrediction,
    OptionModelState,
    choose_planner_goals,
    create_option_model,
    option_model_loss,
    plan_path_scores,
    sample_from_scores,
    update_option_model,
)
from craftax.sea_planner.replay import OptionReplayBuffer, OptionTransition, option_update_count
from craftax.sea_planner.train import build_parser, make_configs, make_env, make_runtime

GRAPH_PATH = Path(__file__).resolve().parents[2] / "craftax_classic_22.npz"


# ---------------------------------------------------------------- toy model


def _toy_state():
    """Goal 1 is nearly impossible until goal 0 is done; goal 2 is an easy leaf.

    The latent is the completed vector, so the toy model is exact.
    """

    def apply_fn(params, *inputs, method=None):
        if method.__name__ == "encode":
            return inputs[1].astype(jnp.float32)
        if method.__name__ == "edge":
            latent, goal = inputs
            p = jnp.where(
                goal == 0, 0.6,
                jnp.where(goal == 2, 0.7, jnp.where(latent[..., 0] > 0, 0.99, 0.01)),
            )
            return EdgePrediction(
                success_logit=jnp.log(p) - jnp.log1p(-p),
                log_duration=jnp.zeros_like(p),
                death_logit=jnp.full_like(p, -10.0),
                completed_logits=jnp.full(p.shape + (3,), -10.0),
                next_latent=jnp.maximum(latent, jax.nn.one_hot(goal, latent.shape[-1])),
            )
        raise AssertionError(method.__name__)

    return OptionModelState(
        TrainState.create(apply_fn=apply_fn, params={}, tx=optax.sgd(0.0)), {}
    )


def _scores(horizon, use_leaf=False, completed=None, prune="none", graph=None, beam=2):
    completed = jnp.zeros((1, 3), dtype=bool) if completed is None else completed
    return plan_path_scores(
        _toy_state(), jnp.zeros((1, 4)), completed, jnp.array([3]),
        jnp.ones((1, 3), dtype=bool), horizon=horizon, beam_width=beam, gamma=1.0,
        use_leaf=use_leaf, max_duration=300.0, graph=graph, prune=prune,
    )[0]


def test_two_hops_value_a_far_goal_that_one_hop_misses():
    one = _scores(1)
    np.testing.assert_allclose(one, [0.6, 0.01, 0.7], atol=1e-5)
    assert int(jnp.argmax(one)) == 2  # greedy: the easy leaf
    two = _scores(2)
    # 0 -> 1 unlocks the hard goal: 0.6 + 0.6 * 0.99 ; 2 -> 0: 0.7 + 0.7 * 0.6.
    np.testing.assert_allclose(two[0], 0.6 + 0.6 * 0.99, atol=1e-5)
    np.testing.assert_allclose(two[2], 0.7 + 0.7 * 0.6, atol=1e-5)
    assert int(jnp.argmax(two)) == 0


def test_completed_goals_score_nothing_and_leaf_term_changes_scores():
    assert float(_scores(1, completed=jnp.array([[False, False, True]]))[2]) == 0.0
    no_leaf, leaf = _scores(1), _scores(1, use_leaf=True)
    # After goal 0: goals 1 (0.99) and 2 (0.7) are new candidates.
    np.testing.assert_allclose(leaf[0], 0.6 + 0.6 * (0.99 + 0.7), atol=1e-5)
    assert not np.allclose(leaf, no_leaf)


def test_budget_overflow_falls_back_to_one_hop():
    config = PlannerConfig(planner_horizon=3, planner_max_decisions=2)
    needs_goal = jnp.array([True, True, True, False])
    goals, overflow = choose_planner_goals(
        jax.random.PRNGKey(0), _toy_state(), jnp.zeros((4, 4)),
        jnp.zeros((4, 3), dtype=bool), jnp.full((4,), 3),
        jnp.ones((4, 3), dtype=bool), needs_goal, config,
    )
    assert goals.shape == (4,) and bool(((goals >= 0) & (goals < 3)).all())
    assert int(overflow.sum()) == 1  # 3 envs need a goal, the budget is 2
    assert not bool(overflow[3])


def test_sampling_is_scale_free():
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    candidates = jnp.ones((22,), dtype=bool)
    shares = []
    for scale in (0.03, 3.0):
        scores = jnp.zeros((22,)).at[5].set(scale).at[6].set(0.8 * scale)
        picks = jax.vmap(lambda k: sample_from_scores(k, scores, candidates, 0.1))(keys)
        shares.append(float((picks == 5).mean()))
    assert abs(shares[0] - shares[1]) < 0.03
    # exp(10 * 0.2) ~ 7.4 : 1 between the two best goals.
    assert 0.8 < shares[0] < 0.92


def test_leaf_prune_uses_prerequisites_only_for_the_leaf_term():
    graph = jnp.zeros((3, 3), dtype=jnp.int8).at[0, 1].set(1)  # 0 -> 1
    unpruned = _scores(1, use_leaf=True)
    pruned = _scores(1, use_leaf=True, prune="leaf", graph=graph)
    # After goal 2 the unready goal 1 (p=0.01) no longer counts in the leaf.
    np.testing.assert_allclose(unpruned[2], 0.7 + 0.7 * (0.6 + 0.01), atol=1e-5)
    np.testing.assert_allclose(pruned[2], 0.7 + 0.7 * 0.6, atol=1e-5)
    np.testing.assert_allclose(pruned[0], unpruned[0], atol=1e-6)  # 1 is ready after 0


def test_all_prune_forbids_unready_goals_at_later_hops():
    graph = jnp.zeros((3, 3), dtype=jnp.int8).at[0, 1].set(1).at[0, 2].set(1)
    score = _scores(2, prune="all", graph=graph, beam=3)
    # From goal 1, hop 2 may only use goal 0 (goals 1 and 2 need goal 0).
    np.testing.assert_allclose(score[1], 0.01 + 0.01 * 0.6, atol=1e-5)
    # With prune "all", hop-1 candidates are restricted to ready goals.
    config = PlannerConfig(planner_horizon=2, planner_prune="all")
    keys = jax.random.split(jax.random.PRNGKey(1), 200)
    goals = jax.vmap(lambda k: choose_planner_goals(
        k, _toy_state(), jnp.zeros((1, 4)), jnp.zeros((1, 3), dtype=bool),
        jnp.array([3]), jnp.ones((1, 3), dtype=bool), jnp.array([True]), config, graph,
    )[0][0])(keys)
    assert bool((goals == 0).all())


# ---------------------------------------------------------- option model


def _batch(key, size=64, goals=22):
    observation = jax.random.uniform(key, (size, 1345))
    goal = jnp.arange(size) % goals
    return OptionTransition(
        observation=observation,
        completed=jnp.zeros((size, goals), dtype=bool),
        current_node=jnp.full((size,), goals),
        goal=goal,
        next_observation=observation[::-1],
        next_completed=jax.nn.one_hot(goal, goals, dtype=bool),
        next_current_node=goal,
        duration=jnp.arange(1, size + 1),
        done=jnp.zeros((size,), dtype=bool),
        success=goal < 5,  # learnable from the goal alone
        died=goal == 7,
    )


def test_supervised_updates_reduce_success_loss():
    config = PlannerConfig(model_hidden_size=32, model_latent_size=16, learning_rate=3e-3)
    batch = _batch(jax.random.PRNGKey(0))
    state = create_option_model(jax.random.PRNGKey(1), batch.observation[:2], 22, config)
    _, (before, _, _) = option_model_loss(
        state.train_state.params, state.target_params, state.train_state.apply_fn, batch
    )
    for _ in range(50):
        state, metrics = update_option_model(state, batch, 0.995)
    _, (after, _, _) = option_model_loss(
        state.train_state.params, state.target_params, state.train_state.apply_fn, batch
    )
    assert all(np.isfinite(float(v)) for v in metrics.values())
    assert float(after["success"]) < 0.5 * float(before["success"])


def test_replay_keeps_rows_and_update_count_uses_recorded_options():
    replay = OptionReplayBuffer.create(4, (1345,), 22)
    batch = _batch(jax.random.PRNGKey(2), size=6)
    replay = replay.add(batch, jnp.array([True, False, True, True, True, True]))
    assert int(replay.size) == 4  # newest five valid rows, capacity 4
    assert set(np.asarray(replay.goal).tolist()) == {2, 3, 4, 5}
    assert int(option_update_count(320, 256, 4.0)) == 5  # round(1280 / 256)
    assert int(option_update_count(10, 256, 4.0)) == 0


# ---------------------------------------------------------- goals / options


def _runtime(**kwargs):
    graph, ids = load_oracle_graph(GRAPH_PATH)
    return PlannerGoalRuntime(graph=jnp.asarray(graph), achievement_ids=ids, **kwargs)


def _flags(*ids):
    return jnp.zeros((1, 22), dtype=bool).at[0, list(ids)].set(True) if ids else jnp.zeros((1, 22), bool)


def _progress(runtime, objective, last, new, events, completed=()):
    state = GoalState(jnp.array([objective]), _flags(*completed), jnp.array([last]))
    new_flags = _flags(*new)
    return advance_goal_progress(
        runtime, state, jnp.array([False]), new_flags.sum(-1).astype(jnp.float32),
        new_flags, _flags(*events),
    )


def test_rewards_first_target_three_nontarget_one_repeat_zero():
    runtime = _runtime()
    assert float(_progress(runtime, 0, 22, [0], [0])[1][0]) == 3.0
    assert float(_progress(runtime, 0, 22, [0, 4], [0, 4])[1][0]) == 4.0
    repeat = _progress(runtime, 0, 22, [], [0], completed=[0])
    assert bool(repeat[2][0]) and float(repeat[1][0]) == 0.0
    assert float(_progress(_runtime(repeat_target_coefficient=1.0), 0, 22, [], [0], [0])[1][0]) == 1.0
    # Without reselection a repeat is not a success.
    assert not bool(_progress(_runtime(allow_completed_goals=False), 0, 22, [], [0], [0])[2][0])


def test_current_node_moves_on_first_time_or_target_only():
    runtime = _runtime()
    incidental = _progress(runtime, 5, 1, [], [0], completed=[0, 1])
    assert int(incidental[0].last_completed[0]) == 1
    commanded = _progress(runtime, 0, 1, [], [0], completed=[0, 1])
    assert int(commanded[0].last_completed[0]) == 0
    first = _progress(runtime, 5, 1, [4], [4], completed=[0, 1])
    assert int(first[0].last_completed[0]) == 4


def test_option_end_labels_timeout_death_and_warmup_censoring():
    goal_done = jnp.array([False, False, True, False])
    duration = jnp.array([300, 10, 20, 10])
    env_done = jnp.array([False, True, False, False])  # env 1 died
    done = high_level_option_done(goal_done, duration, env_done, 300)
    assert done.tolist() == [True, True, True, False]
    success = goal_done & (duration <= 300)
    assert success.tolist() == [False, False, True, False]  # timeout / death fail
    # Warmup: env 3 is replaced by a non-target first-time achievement.
    events = jnp.array([0.0, 0.0, 1.0, 1.0])
    warm_done = warmup_option_done(goal_done, events, env_done)
    recorded = warmup_option_recorded(warm_done, goal_done, duration, env_done, 300)
    assert warm_done.tolist() == [False, True, True, True]
    assert recorded.tolist() == [False, True, True, False]


def test_candidates_allow_unready_and_completed_goals():
    graph = jnp.zeros((3, 3), dtype=jnp.int8).at[0, 1].set(1)
    completed = jnp.array([[False, False, True]])
    strict = get_goal_candidates(completed, graph)
    assert strict.tolist() == [[True, False, False]]
    loose = get_goal_candidates(completed, graph, allow_unready_goals=True,
                                allow_completed_goals=True)
    assert loose.tolist() == [[True, True, True]]


def test_scores_from_counts_reports_score22_and_score21():
    counts = np.full(22, 50.0)
    counts[11] = 0.0  # eat_plant never succeeds
    scores = scores_from_counts(counts, 100)
    assert len(NO_EAT_PLANT_INDICES) == 21
    assert scores["crafter_score21"] > scores["crafter_score22"]
    np.testing.assert_allclose(scores["crafter_score21"], 50.0, rtol=1e-6)


# ------------------------------------------------------------ CLI / training


def test_cli_flags_reach_configs():
    args = build_parser().parse_args([
        "--output", "unused", "--high-level-controller", "graph-planner",
        "--planner-horizon", "3", "--planner-leaf", "false", "--planner-beam-width", "5",
        "--planner-max-decisions", "64", "--planner-gamma", "0.95",
        "--planner-temperature", "0.2", "--planner-prune", "leaf",
        "--planner-prune-width", "7", "--model-hidden-size", "128",
        "--model-latent-size", "64", "--model-target-ema", "0.99",
        "--high-level-q-learning-rate", "1e-4", "--high-level-q-replay-ratio", "2",
        "--high-level-goal-timeout", "200", "--allow-unready-goals", "false",
        "--allow-completed-goals", "false", "--goal-success-coefficient", "3",
        "--non-target-achievement-coefficient", "1", "--repeat-target-coefficient", "0.5",
        "--dynamic-warmup-steps", "1000",
    ])
    _, planner = make_configs(args)
    assert (planner.planner_horizon, planner.planner_leaf, planner.planner_beam_width) == (3, False, 5)
    assert (planner.planner_max_decisions, planner.planner_gamma) == (64, 0.95)
    assert (planner.planner_temperature, planner.planner_prune, planner.planner_prune_width) == (0.2, "leaf", 7)
    assert (planner.model_hidden_size, planner.model_latent_size, planner.model_target_ema) == (128, 64, 0.99)
    assert (planner.learning_rate, planner.replay_ratio, planner.goal_timeout) == (1e-4, 2.0, 200)
    assert not planner.allow_unready_goals and not planner.allow_completed_goals
    assert planner.warmup_steps == 1000
    runtime = make_runtime(vars(args), GRAPH_PATH)
    assert runtime.repeat_target_coefficient == 0.5 and not runtime.allow_completed_goals
    defaults = build_parser().parse_args(["--output", "unused"])
    env = make_env(vars(defaults))
    assert env.mortal and env.combat == "org" and env.idle_timeout == 10_000 and env.symbolic
    _, planner = make_configs(defaults)
    assert planner.planner_horizon == 2 and planner.allow_completed_goals
    assert planner.warmup_steps == 50_000_000


def _tiny(warmup_steps=0, goal_timeout=300, **planner_kwargs):
    from craftax.sea_planner.ppo import make_train

    args = vars(build_parser().parse_args(["--output", "unused"]))
    config = PPOConfig(total_timesteps=8 * 16 * 3, num_envs=8, num_steps=16,
                       update_epochs=1, num_minibatches=2, hidden_size=32, reset_ratio=2)
    planner = PlannerConfig(
        warmup_steps=warmup_steps, replay_capacity=64, batch_size=8, min_replay_size=8,
        goal_timeout=goal_timeout, model_hidden_size=16, model_latent_size=8,
        planner_max_decisions=4, **planner_kwargs,
    )
    return make_train, config, make_env(args), make_runtime(args, GRAPH_PATH), planner


def test_ppo_transition_uses_the_goal_followed_in_that_step():
    make_train, config, env, runtime, planner = _tiny(goal_timeout=1)
    result = jax.jit(make_train(config, env, runtime, planner, return_rollout=True))(
        jax.random.PRNGKey(0)
    )
    followed, options = result["rollout"]
    followed = np.asarray(followed).reshape(-1, config.num_envs)
    selected = np.asarray(options.selected_goal).reshape(-1, config.num_envs)
    ended = np.asarray(options.option_done).reshape(-1, config.num_envs)
    assert ended.all()  # timeout 1: every option ends after one step
    # The goal followed at step t is the one selected at the end of step t-1.
    np.testing.assert_array_equal(followed[1:], selected[:-1])
    assert (followed[1:] != selected[1:]).any()
    assert int(result["metrics"]["high_level_update_count"].sum()) > 0
    assert np.isfinite(np.asarray(result["metrics"]["loss"])).all()


def test_warmup_uses_meta_controller_and_censors_options():
    make_train, config, env, runtime, planner = _tiny(warmup_steps=10**9)
    result = jax.jit(make_train(config, env, runtime, planner, return_rollout=True))(
        jax.random.PRNGKey(1)
    )
    _, options = result["rollout"]
    ended = np.asarray(options.option_done)
    recorded = np.asarray(options.recorded)
    assert np.asarray(options.warmup).all()
    assert not (recorded & ~ended).any()
    assert float(np.asarray(result["metrics"]["planner_overflow_fraction"]).sum()) == 0.0


def test_frozen_mode_changes_no_weights():
    make_train, config, env, runtime, planner = _tiny()
    trained = jax.jit(make_train(config, env, runtime, planner))(jax.random.PRNGKey(2))
    frozen = {
        "policy": trained["runner_state"][0].params,
        "option_model": trained["runner_state"][7].model.train_state.params,
        "option_model_target": trained["runner_state"][7].model.target_params,
    }
    result = jax.jit(make_train(config, env, runtime, planner, frozen_params=frozen))(
        jax.random.PRNGKey(84001)
    )
    after = result["runner_state"]
    for before, now in (
        (frozen["policy"], after[0].params),
        (frozen["option_model"], after[7].model.train_state.params),
        (frozen["option_model_target"], after[7].model.target_params),
    ):
        same = jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), before, now)
        assert all(jax.tree.leaves(same))
    assert int(result["metrics"]["high_level_update_count"].sum()) == 0
