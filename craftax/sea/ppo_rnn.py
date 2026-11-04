"""Pure-JAX recurrent PPO backbone for SEA Craftax-Classic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from craftax.sea.dynamic_graph import (
    HighLevelReplayBuffer,
    HighLevelTransitionBatch,
    compute_dynamic_edge_scores,
    compute_q_update_count,
    evaluate_high_level_q,
    get_ready_goal_candidates,
    high_level_option_done,
    initialize_high_level_q,
    select_high_level_goal,
    update_high_level_q,
)
from craftax.sea.goal import (
    GoalState,
    advance_goal_progress,
    apply_goal_transition,
    initialize_goal_state,
)
from craftax.sea.metrics import rollout_episode_metrics
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
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    hidden_size: int = 256
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
    achievements: jax.Array
    episode_length: jax.Array


class DynamicStepMetrics(NamedTuple):
    ended: jax.Array
    success: jax.Array
    duration: jax.Array
    selected_goal: jax.Array
    uniform_selection: jax.Array


@struct.dataclass
class DynamicTrainingState:
    q_state: object
    replay: HighLevelReplayBuffer
    start_observation: jax.Array
    start_completed: jax.Array
    start_current_node: jax.Array
    discounted_reward: jax.Array
    duration: jax.Array
    global_env_steps: jax.Array


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


def make_train(
    config: PPOConfig,
    env,
    *,
    pixel: bool = False,
    goal_runtime=None,
    dynamic_config=None,
    progress_callback=None,
    progress_interval: int = 50,
):
    """Build a jittable base or joint goal-policy training function."""

    config.validate()
    if progress_interval < 1:
        raise ValueError("progress_interval must be positive")
    if dynamic_config is not None:
        dynamic_config.validate()
        if pixel or goal_runtime is None:
            raise ValueError(
                "dynamic graph training requires symbolic observations and a goal runtime"
            )
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

        dynamic_state = None

        def sample_goals(selection_key, candidates, scores, use_uniform):
            keys = jax.random.split(selection_key, config.num_envs)
            selected, probabilities = jax.vmap(
                select_high_level_goal, in_axes=(0, 0, 0, None, None)
            )(
                keys,
                candidates,
                scores,
                dynamic_config.temperature,
                use_uniform,
            )
            return selected, probabilities

        def choose_uniform_goals(selection_key, completed):
            candidates = get_ready_goal_candidates(
                completed, goal_runtime.graph, goal_runtime.include_new_tasks
            )
            scores = jnp.zeros(candidates.shape, dtype=jnp.float32)
            return sample_goals(
                selection_key, candidates, scores, jnp.asarray(True)
            )

        def choose_dynamic_goals(
            selection_key, current_observation, completed, current_node, q_state
        ):
            full_values, base_values = evaluate_high_level_q(
                q_state, current_observation, completed, current_node
            )
            candidates = get_ready_goal_candidates(
                completed, goal_runtime.graph, goal_runtime.include_new_tasks
            )
            scores = compute_dynamic_edge_scores(
                goal_runtime.graph,
                current_node,
                full_values,
                base_values,
                dynamic_config.residual_weight,
            )
            return sample_goals(
                selection_key, candidates, scores, jnp.asarray(False)
            )

        if goal_runtime is None:
            goal_state = None
        elif dynamic_config is None:
            key, goal_key = jax.random.split(key)
            goal_state = initialize_goal_state(
                goal_key,
                config.num_envs,
                goal_runtime.graph,
                goal_runtime.include_new_tasks,
            )
        else:
            key, q_key, goal_key = jax.random.split(key, 3)
            completed = jnp.zeros(
                (config.num_envs, num_objectives), dtype=bool
            )
            current_node = jnp.full(
                (config.num_envs,), num_objectives, dtype=jnp.int32
            )
            q_state = initialize_high_level_q(
                q_key,
                observation,
                completed,
                current_node,
                num_objectives,
                config.hidden_size,
                dynamic_config.learning_rate,
            )
            if dynamic_config.warmup_steps > 0:
                objective, _ = choose_uniform_goals(goal_key, completed)
            else:
                objective, _ = choose_dynamic_goals(
                    goal_key, observation, completed, current_node, q_state
                )
            goal_state = GoalState(objective, completed, current_node)
            dynamic_state = DynamicTrainingState(
                q_state=q_state,
                replay=HighLevelReplayBuffer.create(
                    dynamic_config.replay_capacity,
                    observation.shape[1:],
                    num_objectives,
                ),
                start_observation=observation,
                start_completed=completed,
                start_current_node=current_node,
                discounted_reward=jnp.zeros(
                    (config.num_envs,), dtype=jnp.float32
                ),
                duration=jnp.zeros((config.num_envs,), dtype=jnp.int32),
                global_env_steps=jnp.asarray(0, dtype=jnp.int32),
            )

        runner_state = (
            train_state,
            env_state,
            observation,
            jnp.zeros((config.num_envs,), dtype=bool),
            hidden,
            goal_state,
            key,
            dynamic_state,
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
                    dynamic_state,
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
                dynamic_step_metrics = None
                if goal_runtime is None:
                    goal_done = jnp.zeros_like(env_done)
                    policy_reward = reward
                    bootstrap_done = env_done
                elif dynamic_config is None:
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
                        info["event_count"],
                        new_achievements=info["new_achievements"],
                        health_reward=reward - info["event_count"],
                    )
                    bootstrap_done = env_done | goal_done
                else:
                    (
                        progress_state,
                        policy_reward,
                        goal_done,
                        _,
                        _,
                        _,
                        transition_completed,
                        transition_current_node,
                    ) = advance_goal_progress(
                        goal_runtime,
                        goal_state,
                        observation,
                        action,
                        info["terminal_obs"],
                        env_done,
                        info["event_count"],
                        new_achievements=info["new_achievements"],
                        health_reward=reward - info["event_count"],
                    )
                    duration = dynamic_state.duration + 1
                    discounted_reward = (
                        dynamic_state.discounted_reward
                        + (config.gamma ** dynamic_state.duration) * reward
                    )
                    option_done = high_level_option_done(
                        goal_done, duration, env_done, dynamic_config.goal_timeout
                    )
                    replay = dynamic_state.replay.add(
                        HighLevelTransitionBatch(
                            observation=dynamic_state.start_observation,
                            completed=dynamic_state.start_completed,
                            current_node=dynamic_state.start_current_node,
                            goal=goal_state.objective,
                            reward=discounted_reward,
                            next_observation=info["terminal_obs"],
                            next_completed=transition_completed,
                            next_current_node=transition_current_node,
                            duration=duration,
                            done=env_done,
                            success=goal_done,
                        ),
                        option_done,
                    )
                    global_env_steps = (
                        dynamic_state.global_env_steps + config.num_envs
                    )
                    use_uniform = (
                        global_env_steps < dynamic_config.warmup_steps
                    )

                    def select_new_goals(_):
                        def uniform(_):
                            return choose_uniform_goals(
                                goal_key, progress_state.completed
                            )[0]

                        def dynamic(_):
                            return choose_dynamic_goals(
                                goal_key,
                                next_observation,
                                progress_state.completed,
                                progress_state.last_completed,
                                dynamic_state.q_state,
                            )[0]

                        return jax.lax.cond(
                            use_uniform, uniform, dynamic, operand=None
                        )

                    selected = jax.lax.cond(
                        jnp.any(option_done),
                        select_new_goals,
                        lambda _: goal_state.objective,
                        operand=None,
                    )
                    objective = jnp.where(
                        option_done, selected, goal_state.objective
                    )
                    goal_state = progress_state.replace(objective=objective)
                    dynamic_state = dynamic_state.replace(
                        replay=replay,
                        start_observation=jnp.where(
                            option_done[:, None],
                            next_observation,
                            dynamic_state.start_observation,
                        ),
                        start_completed=jnp.where(
                            option_done[:, None],
                            progress_state.completed,
                            dynamic_state.start_completed,
                        ),
                        start_current_node=jnp.where(
                            option_done,
                            progress_state.last_completed,
                            dynamic_state.start_current_node,
                        ),
                        discounted_reward=jnp.where(
                            option_done, 0.0, discounted_reward
                        ),
                        duration=jnp.where(option_done, 0, duration),
                        global_env_steps=global_env_steps,
                    )
                    bootstrap_done = env_done | option_done
                    dynamic_step_metrics = DynamicStepMetrics(
                        ended=option_done,
                        success=goal_done,
                        duration=duration,
                        selected_goal=selected,
                        uniform_selection=jnp.full(
                            option_done.shape, use_uniform
                        ),
                    )
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
                    info["achievements"],
                    info["episode_length"],
                )
                step_output = (
                    (transition, dynamic_step_metrics)
                    if dynamic_config is not None
                    else transition
                )
                return (
                    train_state,
                    env_state,
                    next_observation,
                    next_reset,
                    hidden,
                    goal_state,
                    key,
                    dynamic_state,
                ), step_output

            runner_state, rollout = jax.lax.scan(
                env_step, runner_state, None, length=config.num_steps
            )
            if dynamic_config is None:
                transitions = rollout
                dynamic_rollout = None
            else:
                transitions, dynamic_rollout = rollout
            (
                train_state,
                env_state,
                observation,
                last_reset,
                hidden,
                goal_state,
                key,
                dynamic_state,
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

            q_metrics = None
            q_update_count = None
            if dynamic_config is not None:
                new_transition_count = dynamic_rollout.ended.astype(
                    jnp.int32
                ).sum()
                requested_q_updates = compute_q_update_count(
                    new_transition_count,
                    dynamic_config.batch_size,
                    dynamic_config.high_level_q_replay_ratio,
                )
                q_update_count = jnp.where(
                    dynamic_state.replay.size
                    >= dynamic_config.min_replay_size,
                    requested_q_updates,
                    0,
                ).astype(jnp.int32)
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                initial_q_metrics = {
                    "full_q_loss": zero,
                    "base_q_loss": zero,
                    "full_q_mean": zero,
                    "full_q_std": zero,
                    "base_q_mean": zero,
                    "base_q_std": zero,
                    "q_residual_mean": zero,
                    "q_residual_std": zero,
                }
                key, q_update_key = jax.random.split(key)

                def q_update(_, loop_state):
                    q_state, sample_key, metric_sums = loop_state
                    sample_key, batch_key = jax.random.split(sample_key)
                    batch = dynamic_state.replay.sample(
                        batch_key, dynamic_config.batch_size
                    )
                    q_state, update_metrics = update_high_level_q(
                        q_state,
                        batch,
                        goal_runtime.graph,
                        goal_runtime.include_new_tasks,
                        config.gamma,
                    )
                    metric_sums = jax.tree.map(
                        lambda total, value: total + value,
                        metric_sums,
                        update_metrics,
                    )
                    return q_state, sample_key, metric_sums

                q_state, _, q_metric_sums = jax.lax.fori_loop(
                    0,
                    q_update_count,
                    q_update,
                    (
                        dynamic_state.q_state,
                        q_update_key,
                        initial_q_metrics,
                    ),
                )
                q_metric_denominator = jnp.maximum(
                    q_update_count.astype(jnp.float32), 1.0
                )
                q_metrics = jax.tree.map(
                    lambda value: value / q_metric_denominator,
                    q_metric_sums,
                )
                dynamic_state = dynamic_state.replace(q_state=q_state)

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
                    gradient_norm = optax.global_norm(gradients)
                    train_state = train_state.apply_gradients(grads=gradients)
                    return train_state, (loss, metrics, gradient_norm)

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
                dynamic_state,
            )
            metrics = {
                "mean_reward": transitions.reward.mean(),
                "episode_end_rate": transitions.env_done.mean(),
                "goal_end_rate": (
                    transitions.bootstrap_done & ~transitions.env_done
                ).mean(),
                "loss": losses[0].mean(),
                "actor_loss": losses[1][0].mean(),
                "value_loss": losses[1][1].mean(),
                "entropy": losses[1][2].mean(),
                "gradient_norm": losses[2].mean(),
                "gradient_clip_fraction": (
                    losses[2] > config.max_grad_norm
                ).mean(),
            }
            if dynamic_config is not None:
                ended = dynamic_rollout.ended
                transition_count = ended.astype(jnp.float32).sum()
                denominator = jnp.maximum(transition_count, 1.0)
                selected_counts = (
                    jax.nn.one_hot(
                        dynamic_rollout.selected_goal, num_objectives
                    )
                    * ended[..., None]
                ).sum(axis=(0, 1))
                uniform_count = (
                    dynamic_rollout.uniform_selection & ended
                ).astype(jnp.float32).sum()
                metrics.update(q_metrics)
                metrics.update(
                    {
                        "high_level_replay_size": dynamic_state.replay.size,
                        "high_level_transition_count": transition_count,
                        "high_level_q_update_count": q_update_count,
                        "selected_goal_distribution": (
                            selected_counts / denominator
                        ),
                        "uniform_selection_fraction": (
                            uniform_count / denominator
                        ),
                        "dynamic_selection_fraction": (
                            (transition_count - uniform_count) / denominator
                        ),
                        "selected_goal_success_rate": (
                            dynamic_rollout.success.astype(jnp.float32).sum()
                            / denominator
                        ),
                        "mean_high_level_duration": (
                            (
                                dynamic_rollout.duration
                                * ended.astype(dynamic_rollout.duration.dtype)
                            ).astype(jnp.float32).sum()
                            / denominator
                        ),
                    }
                )
            metrics.update(rollout_episode_metrics(transitions))
            if progress_callback is not None:
                update_count = train_state.step // (
                    config.update_epochs * config.num_minibatches
                )

                checkpoint_params = train_state.params
                if dynamic_config is not None:
                    checkpoint_params = {
                        "policy": train_state.params,
                        "full_q": dynamic_state.q_state.full_q.params,
                        "base_q": dynamic_state.q_state.base_q.params,
                        "target_full_q": (
                            dynamic_state.q_state.target_full_params
                        ),
                    }

                def report(_):
                    jax.debug.callback(
                        progress_callback,
                        update_count,
                        metrics,
                        checkpoint_params,
                        ordered=True,
                    )

                jax.lax.cond(
                    (update_count == 1)
                    | (update_count % progress_interval == 0)
                    | (update_count == num_updates),
                    report, lambda _: None, operand=None,
                )
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
