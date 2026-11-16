import numpy as np
import pytest

from craftax.sea.evaluation import score_rates, summarize_episodes


def test_score_has_correct_percentage_scale_and_retains_zero_achievements():
    assert score_rates(np.zeros(22)) == 0
    assert score_rates(np.ones(22)) == pytest.approx(100)
    easy_only = np.ones((20, 22), bool)
    easy_only[:, [8, 19, 20, 21]] = False
    summary = summarize_episodes(easy_only, np.ones(20, bool))
    assert summary["crafter_score22"] == pytest.approx(np.expm1(18/22*np.log(101)))
    assert summary["crafter_score22"] < 43
    assert summary["hard4_rate"] == 0
    assert len(summary["achievement_success_rate"]) == 22


def test_all_episodes_share_denominator_even_when_they_unlock_nothing():
    achievements = np.zeros((10, 22), bool)
    achievements[0, 0] = True
    summary = summarize_episodes(achievements, np.ones(10, bool))
    assert summary["achievement_success_rate"]["collect_wood"] == .1
    assert summary["achievements_per_episode"] == .1
    assert summary["num_episodes"] == 10
    assert summary["crafter_score22"] == pytest.approx(np.expm1(np.log(11)/22))


def test_unfinished_and_repeat_counted_episodes_are_rejected():
    with pytest.raises(ValueError, match="every predetermined"):
        summarize_episodes(np.zeros((2, 22), bool), [True, False])
    with pytest.raises(ValueError, match="once per episode"):
        summarize_episodes(np.full((2, 22), 2), [True, True])


def test_fixed_cohort_counts_short_empty_and_long_successful_episodes_once(monkeypatch):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from flax import struct
    import craftax.sea.evaluation as evaluation
    from craftax.sea.dynamic_graph import initialize_high_level_q
    from craftax.sea.networks import ActorCriticRNN, ScannedGRU

    @struct.dataclass
    class State:
        timestep: object
        duration: object
        achievements: object

    @struct.dataclass
    class Params:
        max_timesteps: int = 4
        always_diamond: bool = True

    class Env:
        num_actions = 2
        idle_timeout = 4
        immortal = False
        health_reward_coef = 0.
        combat_damage_mode = "lethal-only"
        default_params = Params()

        def reset(self, key, params):
            duration = 1 + 2 * jax.random.randint(key, (), 0, 2)
            return jnp.zeros(1345), State(jnp.int32(0), duration, jnp.zeros(22, bool))

        def step(self, key, state, action, params):
            t = state.timestep + 1
            achievements = state.achievements.at[0].set(t >= 3)
            novel = achievements & ~state.achievements
            done = t >= state.duration
            info = dict(event_count=novel.sum(), new_achievements=novel,
                achievement_events=novel, achievements=achievements, episode_length=t,
                health_done=jnp.asarray(False), idle_done=jnp.asarray(False), time_done=done)
            return jnp.zeros(1345), State(t, state.duration, achievements), novel.sum(), done, info

    monkeypatch.setattr(evaluation, "make_sea_craftax_classic_env", lambda **kwargs: Env())
    config = dict(args=dict(pixels=False, no_new_tasks=True, idle_timeout=4,
        immortal=False, health_reward_coef=0., combat_damage_mode="lethal-only"),
        goal_ppo=dict(hidden_size=8), dynamic_graph=dict(allow_unready_goals=True,
        allow_completed_goals=False, temperature=1., goal_timeout=2))
    oracle = SimpleNamespace(graph=np.zeros((22, 22)), achievement_ids=tuple(range(22)))
    initialize, rollout, _ = evaluation.build_evaluator(config, oracle, episodes=8, block=4)
    state, _ = initialize(1_000_000, 314159)
    expected_lengths = np.asarray(state.env_state.duration)
    assert set(expected_lengths) == {1, 3}
    q = initialize_high_level_q(jax.random.PRNGKey(4), state.observation,
        state.goals.completed, state.goals.last_completed, 22, 8, 1e-3)
    policy = ActorCriticRNN(2, 8, False, 22)
    params = policy.init(jax.random.PRNGKey(5), ScannedGRU.initialize_carry(8, 8),
        state.observation[None], state.rnn_reset[None], state.goals.objective[None], state.goals.completed[None])
    state = rollout(state, params, q.full_q.params, q.base_q.params, jnp.float32(0), jnp.float32(0), jnp.int32(0))
    # Running a further block must never count an already-finished slot again.
    state = rollout(state, params, q.full_q.params, q.base_q.params, jnp.float32(0), jnp.float32(0), jnp.int32(0))
    np.testing.assert_array_equal(state.terminal_count, np.ones(8))
    np.testing.assert_array_equal(state.lengths, expected_lengths)
    np.testing.assert_array_equal(state.achievements[:, 0], expected_lengths == 3)
    result = summarize_episodes(state.achievements, ~np.asarray(state.active))
    assert result["num_episodes"] == 8
    assert result["achievement_success_rate"]["collect_wood"] == np.mean(expected_lengths == 3)
    # Full-Q evaluation must follow Q_F, even when Q_B and the residual favor
    # other goals. Saturated logits make this routing check deterministic.
    full = jax.tree.map(jnp.zeros_like, q.full_q.params)
    base = jax.tree.map(jnp.zeros_like, q.base_q.params)
    full["params"]["Dense_2"]["bias"] = full["params"]["Dense_2"]["bias"].at[7].set(1000.)
    base["params"]["Dense_2"]["bias"] = base["params"]["Dense_2"]["bias"].at[7].set(2000.).at[8].set(3000.)
    state, _ = initialize(1_000_000, 314159)
    state = rollout(state, params, full, base, jnp.float32(.2), jnp.float32(1), jnp.int32(2))
    selections = np.asarray(state.selected_goal_counts)
    assert (selections[:, 7] >= 1).all()
    assert np.delete(selections, 7, axis=1).sum() == 0
