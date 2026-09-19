import jax
import jax.numpy as jnp
import numpy as np

from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime
from craftax.sea.metrics import rollout_episode_metrics, summarize_training_metrics
from craftax.sea.networks import TransitionEncoder
from craftax.sea.ppo_rnn import PPOConfig, Transition, calculate_gae, make_train
from craftax.sea.rollout import collect_transition_dataset
from craftax.sea.train import build_parser
from craftax.sea.vector_env import SeaAutoResetVecEnv


def test_vector_env_preserves_terminal_observation():
    env = make_sea_craftax_classic_env(idle_timeout=1)
    vec = SeaAutoResetVecEnv(env, num_envs=2, reset_ratio=1)
    params = env.default_params
    observation, state = vec.reset(jax.random.PRNGKey(0), params)
    next_observation, _, _, done, info = vec.step(
        jax.random.PRNGKey(1), state, jnp.zeros((2,), dtype=jnp.int32), params
    )
    assert done.all()
    assert info["terminal_obs"].shape == observation.shape
    assert next_observation.shape == observation.shape


def test_training_cli_defaults_to_pixels():
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.pixels is True
    assert defaults.hidden_size == 256
    assert defaults.embedding_size == 256
    assert defaults.discovery_interactions == 50_000_000
    assert defaults.contrast_interactions == 1_000_000
    assert defaults.clustering_transitions == 10_000
    assert defaults.discovery_batch_size == 2560
    assert defaults.discovery_num_envs == 32
    assert parser.parse_args(["--symbolic-debug"]).pixels is False


def test_goal_boundary_cuts_gae_bootstrap():
    transition = Transition(
        rnn_reset=jnp.array([[False]]),
        bootstrap_done=jnp.array([[True]]),
        env_done=jnp.array([[False]]),
        action=jnp.array([[0]]),
        value=jnp.array([[2.0]]),
        reward=jnp.array([[1.0]]),
        log_prob=jnp.array([[0.0]]),
        observation=jnp.zeros((1, 1, 2)),
        next_observation=jnp.zeros((1, 1, 2)),
        objective=jnp.array([[0]]),
        completed=jnp.zeros((1, 1, 1), dtype=bool),
        achievements=jnp.zeros((1, 1, 22), dtype=bool),
        episode_length=jnp.array([[1]]),
    )
    config = PPOConfig(
        total_timesteps=1, num_envs=1, num_steps=1, num_minibatches=1, reset_ratio=1
    )
    advantage, _ = calculate_gae(transition, jnp.array([100.0]), config)
    assert jnp.allclose(advantage, -1.0)


def test_episode_metrics_use_finished_episodes_as_denominator():
    achievements = jnp.zeros((2, 2, 22), dtype=bool)
    achievements = achievements.at[0, 0, 0].set(True)
    achievements = achievements.at[0, 0, 19].set(True)
    achievements = achievements.at[1, 1, 0].set(True)
    transition = Transition(
        rnn_reset=jnp.zeros((2, 2), dtype=bool),
        bootstrap_done=jnp.array([[True, False], [False, True]]),
        env_done=jnp.array([[True, False], [False, True]]),
        action=jnp.zeros((2, 2), dtype=jnp.int32),
        value=jnp.zeros((2, 2)),
        reward=jnp.zeros((2, 2)),
        log_prob=jnp.zeros((2, 2)),
        observation=jnp.zeros((2, 2, 1)),
        next_observation=jnp.zeros((2, 2, 1)),
        objective=jnp.zeros((2, 2), dtype=jnp.int32),
        completed=jnp.zeros((2, 2, 1), dtype=bool),
        achievements=achievements,
        episode_length=jnp.array([[10, 0], [0, 20]]),
    )
    metrics = rollout_episode_metrics(transition)
    assert metrics["episode_count"] == 2
    assert metrics["mean_episode_return"] == 1.5
    assert metrics["mean_episode_length"] == 15
    assert metrics["achievement_success_count"][0] == 2
    assert metrics["achievement_success_count"][19] == 1

    scanned = {name: value[None] for name, value in metrics.items()}
    scanned["loss"] = jnp.array([0.25])
    summary = summarize_training_metrics(scanned)
    assert summary["achievement_success_rate"]["collect_wood"] == 1.0
    assert summary["achievement_success_rate"]["collect_diamond"] == 0.5
    assert summary["episode_count"] == 2


def test_symbolic_ppo_smoke_update_is_finite():
    env = make_sea_craftax_classic_env(symbolic=True, idle_timeout=8)
    config = PPOConfig(
        total_timesteps=16,
        num_envs=2,
        num_steps=4,
        update_epochs=1,
        num_minibatches=1,
        hidden_size=32,
        reset_ratio=1,
    )
    reported_updates = []

    def report(update, metrics, params):
        reported_updates.append(int(update))
        assert np.isfinite(metrics["loss"])
        assert params

    result = jax.jit(make_train(
        config, env, pixel=False, progress_callback=report, progress_interval=2,
    ))(jax.random.PRNGKey(7))
    jax.block_until_ready(result["metrics"])
    jax.effects_barrier()
    assert reported_updates == [1, 2]
    assert jnp.isfinite(result["metrics"]["loss"]).all()

    dataset = collect_transition_dataset(
        env,
        result["runner_state"][0].params,
        config,
        interactions=8,
        seed=8,
        chunk_steps=2,
    )
    assert dataset.new_achievements.shape == (len(dataset), 22)
    np.testing.assert_array_equal(dataset.new_achievements.sum(axis=1), dataset.event_count)
    assert len(dataset) == 8
    assert dataset.observation.dtype == jnp.float16
    assert dataset.next_observation.shape == dataset.observation.shape

    sampled_dataset = collect_transition_dataset(
        env,
        result["runner_state"][0].params,
        config,
        interactions=8,
        seed=9,
        chunk_steps=2,
        max_random_transitions=4,
        max_positive_transitions=0,
    )
    assert sampled_dataset.new_achievements.shape == (len(sampled_dataset), 22)
    np.testing.assert_array_equal(
        sampled_dataset.new_achievements.sum(axis=1), sampled_dataset.event_count
    )
    assert len(sampled_dataset) == 4
    assert sampled_dataset.observation.dtype == jnp.float16


def test_pixel_ppo_smoke_update_is_finite():
    env = make_sea_craftax_classic_env(symbolic=False, idle_timeout=4)
    config = PPOConfig(
        total_timesteps=4,
        num_envs=2,
        num_steps=2,
        update_epochs=1,
        num_minibatches=1,
        hidden_size=16,
        reset_ratio=1,
    )
    result = jax.jit(make_train(config, env, pixel=True))(jax.random.PRNGKey(9))
    assert jnp.isfinite(result["metrics"]["loss"]).all()


def test_goal_conditioned_ppo_smoke_update_is_finite():
    env = make_sea_craftax_classic_env(symbolic=True, idle_timeout=8)
    observation, _ = env.reset(jax.random.PRNGKey(8))
    encoder = TransitionEncoder(hidden_size=16, embedding_size=4, pixel=False)
    encoder_params = encoder.init(
        jax.random.PRNGKey(9),
        observation[None],
        jnp.zeros((1,), dtype=jnp.int32),
        observation[None],
        jnp.zeros((1,), dtype=bool),
    )
    runtime = GoalRuntime(
        transition_network=encoder,
        transition_params=encoder_params,
        centroids=jnp.zeros((2, 4), dtype=jnp.float32),
        threshold=1.0,
        graph=jnp.zeros((2, 2), dtype=jnp.int8),
    )
    config = PPOConfig(
        total_timesteps=8,
        num_envs=2,
        num_steps=4,
        update_epochs=1,
        num_minibatches=1,
        hidden_size=32,
        reset_ratio=1,
    )
    result = jax.jit(make_train(config, env, pixel=False, goal_runtime=runtime))(
        jax.random.PRNGKey(10)
    )
    assert jnp.isfinite(result["metrics"]["loss"]).all()
