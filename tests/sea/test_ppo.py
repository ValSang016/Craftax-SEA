import jax
import jax.numpy as jnp

from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import GoalRuntime
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
    assert parser.parse_args([]).pixels is True
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
    )
    config = PPOConfig(
        total_timesteps=1, num_envs=1, num_steps=1, num_minibatches=1, reset_ratio=1
    )
    advantage, _ = calculate_gae(transition, jnp.array([100.0]), config)
    assert jnp.allclose(advantage, -1.0)


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
    result = jax.jit(make_train(config, env, pixel=False))(jax.random.PRNGKey(7))
    assert jnp.isfinite(result["metrics"]["loss"]).all()

    dataset = collect_transition_dataset(
        env,
        result["runner_state"][0].params,
        config,
        interactions=8,
        seed=8,
        chunk_steps=2,
    )
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
