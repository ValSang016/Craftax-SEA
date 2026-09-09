"""Pure-JAX recurrent PPO backbone for SEA Craftax-Classic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from craftax.sea.goal import (
    apply_goal_transition,
    initialize_goal_state,
)
from craftax.sea.networks import ActorCriticRNN, ScannedGRU
from craftax.sea.vector_env import SeaAutoResetVecEnv


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 1_000_000
    num_envs: int = 64
    num_steps: int = 64
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    hidden_size: int = 512
    reset_ratio: int = 16
    anneal_learning_rate: bool = True

    def validate(self):
        batch_size = self.num_envs * self.num_steps
        if self.total_timesteps < batch_size:
            raise ValueError("total_timesteps must cover at least one rollout")
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_minibatches must divide num_envs")
        if self.num_envs % self.reset_ratio:
            raise ValueError("reset_ratio must divide num_envs")


class Transition(NamedTuple):
    rnn_reset: jax.Array
    bootstrap_done: jax.Array
    env_done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    observation: jax.Array
    next_observation: jax.Array
    objective: jax.Array
    completed: jax.Array


def categorical_log_prob(logits, action):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]


def categorical_entropy(logits):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -(probs * log_probs).sum(axis=-1)


def calculate_gae(transitions, last_value, config: PPOConfig):
    def step(carry, transition):
        gae, next_value = carry
        not_done = 1.0 - transition.bootstrap_done.astype(jnp.float32)
        delta = (
            transition.reward + config.gamma * next_value * not_done - transition.value
        )
        gae = delta + config.gamma * config.gae_lambda * not_done * gae
        return (gae, transition.value), gae

    (_, _), advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        transitions,
        reverse=True,
    )
    return advantages, advantages + transitions.value


def make_train(config: PPOConfig, env, *, pixel: bool = False, goal_runtime=None):
    """Build a jittable base-policy PPO training function.

    Goal-conditioned reward and pseudo-terminal handling are layered on this
    same rollout format by :mod:`craftax.sea.goal_ppo`.
    """

    config.validate()
    vec_env = SeaAutoResetVecEnv(env, config.num_envs, config.reset_ratio)
    params = env.default_params
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    action_dim = env.action_space(params).n
    num_objectives = 0 if goal_runtime is None else goal_runtime.num_objectives
    completed_width = max(1, num_objectives)

    def learning_rate(count):
        if not config.anneal_learning_rate:
            return config.learning_rate
        updates_done = count // (config.num_minibatches * config.update_epochs)
        fraction = 1.0 - updates_done / max(1, num_updates)
        return config.learning_rate * fraction

    def train(key):
        network = ActorCriticRNN(
            action_dim=action_dim,
            hidden_size=config.hidden_size,
            pixel=pixel,
            num_objectives=num_objectives,
        )
        key, init_key, reset_key = jax.random.split(key, 3)
        observation, env_state = vec_env.reset(reset_key, params)
        hidden = ScannedGRU.initialize_carry(config.num_envs, config.hidden_size)
        dummy_reset = jnp.zeros((1, config.num_envs), dtype=bool)
        dummy_objective = jnp.zeros((1, config.num_envs), dtype=jnp.int32)
        dummy_completed = jnp.zeros((1, config.num_envs, completed_width), dtype=bool)
        network_params = network.init(
            init_key,
            hidden,
            observation[None],
            dummy_reset,
            dummy_objective,
            dummy_completed,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=optimizer
        )

        if goal_runtime is None:
            goal_state = None
        else:
            key, goal_key = jax.random.split(key)
            goal_state = initialize_goal_state(
                goal_key,
                config.num_envs,
                goal_runtime.graph,
                goal_runtime.include_new_tasks,
            )

        runner_state = (
            train_state,
            env_state,
            observation,
            jnp.zeros((config.num_envs,), dtype=bool),
            hidden,
            goal_state,
            key,
        )

        def update(runner_state, _):
            initial_hidden = runner_state[4]

            def env_step(runner_state, _):
                (
                    train_state,
                    env_state,
                    observation,
                    last_reset,
                    hidden,
                    goal_state,
                    key,
                ) = runner_state
                key, action_key, step_key, goal_key = jax.random.split(key, 4)
                if goal_runtime is None:
                    objective = jnp.zeros((config.num_envs,), dtype=jnp.int32)
                    completed = jnp.zeros(
                        (config.num_envs, completed_width), dtype=bool
                    )
                else:
                    objective = goal_state.objective
                    completed = goal_state.completed
                hidden, logits, value = network.apply(
                    train_state.params,
                    hidden,
                    observation[None],
                    last_reset[None],
                    objective[None],
                    completed[None],
                )
                action = jax.random.categorical(action_key, logits[0])
                log_prob = categorical_log_prob(logits[0], action)
                next_observation, env_state, reward, env_done, info = vec_env.step(
                    step_key, env_state, action, params
                )
                if goal_runtime is None:
                    goal_done = jnp.zeros_like(env_done)
                    policy_reward = reward
                else:
                    (
                        goal_state,
                        policy_reward,
                        goal_done,
                        _,
                        _,
                    ) = apply_goal_transition(
                        goal_runtime,
                        goal_state,
                        goal_key,
                        observation,
                        action,
                        info["terminal_obs"],
                        env_done,
                        reward,
                    )
                bootstrap_done = env_done | goal_done
                next_reset = (
                    bootstrap_done
                    if goal_runtime is None or goal_runtime.reset_rnn_on_goal
                    else env_done
                )
                transition = Transition(
                    last_reset,
                    bootstrap_done,
                    env_done,
                    action,
                    value[0],
                    policy_reward,
                    log_prob,
                    observation,
                    info["terminal_obs"],
                    objective,
                    completed,
                )
                return (
                    train_state,
                    env_state,
                    next_observation,
                    next_reset,
                    hidden,
                    goal_state,
                    key,
                ), transition

            runner_state, transitions = jax.lax.scan(
                env_step, runner_state, None, length=config.num_steps
            )
            (
                train_state,
                env_state,
                observation,
                last_reset,
                hidden,
                goal_state,
                key,
            ) = runner_state
            if goal_runtime is None:
                objective = jnp.zeros((1, config.num_envs), dtype=jnp.int32)
                completed = jnp.zeros((1, config.num_envs, completed_width), dtype=bool)
            else:
                objective = goal_state.objective[None]
                completed = goal_state.completed[None]
            _, _, last_value = network.apply(
                train_state.params,
                hidden,
                observation[None],
                last_reset[None],
                objective,
                completed,
            )
            advantages, targets = calculate_gae(transitions, last_value[0], config)

            def epoch(update_state, _):
                train_state, key = update_state
                key, permutation_key = jax.random.split(key)
                permutation = jax.random.permutation(permutation_key, config.num_envs)
                shuffled_hidden = initial_hidden[permutation]
                shuffled = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1),
                    (transitions, advantages, targets),
                )
                envs_per_minibatch = config.num_envs // config.num_minibatches
                hidden_batches = shuffled_hidden.reshape(
                    (config.num_minibatches, envs_per_minibatch, config.hidden_size)
                )

                def split_minibatches(x):
                    x = x.reshape(
                        (config.num_steps, config.num_minibatches, envs_per_minibatch)
                        + x.shape[2:]
                    )
                    return jnp.swapaxes(x, 0, 1)

                transition_batches, advantage_batches, target_batches = jax.tree.map(
                    split_minibatches, shuffled
                )

                def minibatch(train_state, batch):
                    initial_h, trajectory, gae, target = batch

                    def loss_fn(network_params):
                        _, logits, value = network.apply(
                            network_params,
                            initial_h,
                            trajectory.observation,
                            trajectory.rnn_reset,
                            trajectory.objective,
                            trajectory.completed,
                        )
                        log_prob = categorical_log_prob(logits, trajectory.action)
                        ratio = jnp.exp(log_prob - trajectory.log_prob)
                        normalized_gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_loss = -jnp.minimum(
                            ratio * normalized_gae,
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_epsilon,
                                1.0 + config.clip_epsilon,
                            )
                            * normalized_gae,
                        ).mean()

                        clipped_value = trajectory.value + jnp.clip(
                            value - trajectory.value,
                            -config.clip_epsilon,
                            config.clip_epsilon,
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - target),
                                jnp.square(clipped_value - target),
                            ).mean()
                        )
                        entropy = categorical_entropy(logits).mean()
                        loss = (
                            actor_loss
                            + config.value_coefficient * value_loss
                            - config.entropy_coefficient * entropy
                        )
                        return loss, (actor_loss, value_loss, entropy)

                    (loss, metrics), gradients = jax.value_and_grad(
                        loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=gradients)
                    return train_state, (loss, metrics)

                train_state, losses = jax.lax.scan(
                    minibatch,
                    train_state,
                    (
                        hidden_batches,
                        transition_batches,
                        advantage_batches,
                        target_batches,
                    ),
                )
                return (train_state, key), losses

            (train_state, key), losses = jax.lax.scan(
                epoch,
                (train_state, key),
                None,
                length=config.update_epochs,
            )
            runner_state = (
                train_state,
                env_state,
                observation,
                last_reset,
                hidden,
                goal_state,
                key,
            )
            metrics = {
                "mean_reward": transitions.reward.mean(),
                "episode_end_rate": transitions.env_done.mean(),
                "goal_end_rate": (
                    transitions.bootstrap_done & ~transitions.env_done
                ).mean(),
                "loss": losses[0].mean(),
            }
            return runner_state, metrics

        runner_state, metrics = jax.lax.scan(
            update, runner_state, None, length=num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


__all__ = [
    "PPOConfig",
    "Transition",
    "calculate_gae",
    "categorical_entropy",
    "categorical_log_prob",
    "make_train",
]
