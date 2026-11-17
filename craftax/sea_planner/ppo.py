"""Goal-conditioned PPO-GRU worker with the supervised graph + K-hop planner.

* worker: the existing recurrent PPO (``craftax.sea.networks``), trained by RL;
* option model F: supervised learning on finished options (no RL);
* goal selection: the existing SEA meta-controller for ``warmup_steps`` env
  steps, then only the planner (softmax over K-hop path scores).

Rules that matter (see tests):
* the PPO transition of a step is conditioned on the goal actually followed
  in that step; a goal chosen at an option end only applies from the next step;
* PPO bootstraps on ``env_done | goal_done`` during warmup (the existing SEA
  rule) and on ``env_done | option_done`` afterwards;
* the option model is updated round(recorded options * ratio / batch) times
  per PPO update once the replay holds ``min_replay_size`` options.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from craftax.sea.goal import GoalState
from craftax.sea.networks import ActorCriticRNN, ScannedGRU
from craftax.sea.ppo_rnn import (
    PPOConfig,
    Transition,
    calculate_gae,
    categorical_entropy,
    categorical_log_prob,
)
from craftax.sea.vector_env import SeaAutoResetVecEnv
from craftax.sea_planner.goals import (
    advance_goal_progress,
    choose_meta_controller_goals,
    get_goal_candidates,
    high_level_option_done,
    normalize_reselection,
    warmup_option_done,
    warmup_option_recorded,
)
from craftax.sea_planner.metrics import episode_metrics
from craftax.sea_planner.option_model import (
    GRAPH_METRIC_NAMES,
    choose_planner_goals,
    epsilon_explore,
    create_option_model,
    update_option_model,
)
from craftax.sea_planner.replay import (
    OptionReplayBuffer,
    OptionTransition,
    option_update_count,
)


@struct.dataclass
class PlannerTrainingState:
    model: object
    replay: OptionReplayBuffer
    start_observation: jax.Array
    start_completed: jax.Array
    start_node: jax.Array
    duration: jax.Array
    env_steps: jax.Array


class OptionStep(NamedTuple):
    option_done: jax.Array
    recorded: jax.Array
    success: jax.Array
    duration: jax.Array
    selected_goal: jax.Array
    selected_new: jax.Array
    overflow: jax.Array
    explored: jax.Array
    warmup: jax.Array


def make_train(
    config: PPOConfig,
    env,
    runtime,
    planner_config,
    *,
    progress_callback=None,
    progress_interval: int = 50,
    metrics_callback=None,
    frozen_params=None,
    return_rollout=False,
):
    """Build a jittable ``train(key)``.

    ``frozen_params={"policy", "option_model", "option_model_target"}`` runs
    the same rollout and goal selection without any PPO or F update.
    ``return_rollout`` (tests only) also returns, per update, the goal each
    PPO transition was conditioned on and the per-step option records.
    """
    config.validate()
    planner_config.validate()
    if progress_interval < 1:
        raise ValueError("progress_interval must be positive")
    if not getattr(env, "symbolic", False):
        raise ValueError("the graph planner needs symbolic observations")
    vec_env = SeaAutoResetVecEnv(env, config.num_envs, config.reset_ratio)
    params = env.default_params
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    action_dim = env.action_space(params).n
    num_goals = runtime.num_goals
    graph = runtime.graph
    frozen = frozen_params is not None
    no_node = jnp.int32(num_goals)

    def learning_rate(count):
        if not config.anneal_learning_rate:
            return config.learning_rate
        updates_done = count // (config.num_minibatches * config.update_epochs)
        return config.learning_rate * (1.0 - updates_done / max(1, num_updates))

    network = ActorCriticRNN(
        action_dim=action_dim,
        hidden_size=config.hidden_size,
        pixel=False,
        num_objectives=num_goals,
    )

    if normalize_reselection(planner_config.allow_completed_goals) != normalize_reselection(
        runtime.allow_completed_goals
    ):
        raise ValueError("planner and goal runtime disagree on goal reselection")
    reselectable = runtime.reselectable

    def planner_goals(key, observation, completed, node, needs_goal, model):
        candidates = get_goal_candidates(
            completed, graph,
            allow_unready_goals=planner_config.allow_unready_goals,
            allow_completed_goals=reselectable,
        )
        # Split only when exploring so epsilon=0 keeps the original RNG stream.
        if planner_config.planner_epsilon > 0:
            plan_key, explore_key = jax.random.split(key)
        else:
            plan_key, explore_key = key, key
        goals, overflow = choose_planner_goals(
            plan_key, model, observation, completed, node, candidates, needs_goal,
            planner_config, graph=graph,
        )
        goals, explored = epsilon_explore(
            explore_key, goals, candidates, planner_config.planner_epsilon
        )
        return goals, overflow, explored

    def train(key):
        key, init_key, reset_key, model_key, goal_key = jax.random.split(key, 5)
        observation, env_state = vec_env.reset(reset_key, params)
        hidden = ScannedGRU.initialize_carry(config.num_envs, config.hidden_size)
        network_params = network.init(
            init_key,
            hidden,
            observation[None],
            jnp.zeros((1, config.num_envs), dtype=bool),
            jnp.zeros((1, config.num_envs), dtype=jnp.int32),
            jnp.zeros((1, config.num_envs, num_goals), dtype=bool),
        )
        if frozen:
            network_params = frozen_params["policy"]
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate, eps=1e-5),
            ),
        )
        model = create_option_model(model_key, observation, num_goals, planner_config)
        if frozen:
            model = model.replace(
                train_state=model.train_state.replace(params=frozen_params["option_model"]),
                target_params=frozen_params["option_model_target"],
            )

        completed = jnp.zeros((config.num_envs, num_goals), dtype=bool)
        node = jnp.full((config.num_envs,), no_node)
        if planner_config.warmup_steps > 0:
            objective = choose_meta_controller_goals(goal_key, completed, node, graph)
        else:
            objective, _, _ = planner_goals(
                goal_key, observation, completed, node,
                jnp.ones((config.num_envs,), dtype=bool), model,
            )
        goal_state = GoalState(objective, completed, node)
        planner_state = PlannerTrainingState(
            model=model,
            replay=OptionReplayBuffer.create(
                planner_config.replay_capacity, observation.shape[1:], num_goals
            ),
            start_observation=observation,
            start_completed=completed,
            start_node=node,
            duration=jnp.zeros((config.num_envs,), dtype=jnp.int32),
            env_steps=jnp.asarray(0, dtype=jnp.int32),
        )
        runner_state = (
            train_state, env_state, observation,
            jnp.zeros((config.num_envs,), dtype=bool), hidden, goal_state, key,
            planner_state,
        )

        def env_step(runner_state, _):
            (train_state, env_state, observation, last_reset, hidden,
             goal_state, key, planner_state) = runner_state
            key, action_key, step_key, select_key = jax.random.split(key, 4)
            # The goal followed in this step.
            objective = goal_state.objective
            completed = goal_state.completed
            hidden, logits, value = network.apply(
                train_state.params, hidden, observation[None], last_reset[None],
                objective[None], completed[None],
            )
            action = jax.random.categorical(action_key, logits[0])
            log_prob = categorical_log_prob(logits[0], action)
            next_observation, env_state, _, env_done, info = vec_env.step(
                step_key, env_state, action, params
            )
            (progress, reward, goal_done, _, end_completed, end_node) = advance_goal_progress(
                runtime, goal_state, env_done, info["event_count"],
                info["new_achievements"], info["achievement_events"],
            )
            reward = reward + info["health_reward"]

            duration = planner_state.duration + 1
            in_warmup = planner_state.env_steps < planner_config.warmup_steps
            timeout = planner_config.goal_timeout
            sea_done = warmup_option_done(goal_done, info["event_count"], env_done)
            option_done = jnp.where(
                in_warmup,
                sea_done,
                high_level_option_done(goal_done, duration, env_done, timeout),
            )
            recorded = jnp.where(
                in_warmup,
                warmup_option_recorded(option_done, goal_done, duration, env_done, timeout),
                option_done,
            )
            success = goal_done & (duration <= timeout)
            replay = planner_state.replay.add(
                OptionTransition(
                    observation=planner_state.start_observation,
                    completed=planner_state.start_completed,
                    current_node=planner_state.start_node,
                    goal=objective,
                    next_observation=info["terminal_obs"],
                    next_completed=end_completed,
                    next_current_node=end_node,
                    duration=duration,
                    done=env_done,
                    success=success,
                    died=info["death_done"],
                ),
                recorded,
            )
            env_steps = planner_state.env_steps + config.num_envs
            select_warmup = env_steps < planner_config.warmup_steps
            no_overflow = jnp.zeros_like(option_done)

            def select(_):
                def warmup(_):
                    goals = choose_meta_controller_goals(
                        select_key, progress.completed, progress.last_completed, graph
                    )
                    return goals, no_overflow, no_overflow

                def plan(_):
                    return planner_goals(
                        select_key, next_observation, progress.completed,
                        progress.last_completed, option_done, planner_state.model,
                    )

                return jax.lax.cond(select_warmup, warmup, plan, operand=None)

            selected, overflow, explored = jax.lax.cond(
                option_done.any(), select,
                lambda _: (objective, no_overflow, no_overflow), operand=None,
            )
            next_objective = jnp.where(option_done, selected, objective)
            selected_new = ~jnp.take_along_axis(
                progress.completed, selected[:, None], axis=1
            )[:, 0]
            goal_state = progress.replace(objective=next_objective)
            planner_state = planner_state.replace(
                replay=replay,
                start_observation=jnp.where(
                    option_done[:, None], next_observation, planner_state.start_observation
                ),
                start_completed=jnp.where(
                    option_done[:, None], progress.completed, planner_state.start_completed
                ),
                start_node=jnp.where(
                    option_done, progress.last_completed, planner_state.start_node
                ),
                duration=jnp.where(option_done, 0, duration),
                env_steps=env_steps,
            )
            bootstrap_done = jnp.where(
                in_warmup, env_done | goal_done, env_done | option_done
            )
            next_reset = bootstrap_done if runtime.reset_rnn_on_goal else env_done
            transition = Transition(
                last_reset, bootstrap_done, env_done, action, value[0], reward,
                log_prob, observation, info["terminal_obs"], objective, completed,
                info["achievements"], info["episode_length"],
            )
            option_step = OptionStep(
                option_done=option_done,
                recorded=recorded,
                success=goal_done & option_done,
                duration=duration,
                selected_goal=selected,
                selected_new=selected_new,
                overflow=overflow & option_done,
                explored=explored & option_done,
                warmup=jnp.broadcast_to(select_warmup, option_done.shape),
            )
            runner_state = (
                train_state, env_state, next_observation, next_reset, hidden,
                goal_state, key, planner_state,
            )
            return runner_state, (transition, option_step)

        def update_model(planner_state, recorded, key):
            requested = option_update_count(
                recorded.sum(), planner_config.batch_size, planner_config.replay_ratio
            )
            count = jnp.where(
                (planner_state.replay.size >= planner_config.min_replay_size) & (not frozen),
                requested, 0,
            ).astype(jnp.int32)
            zero = jnp.asarray(0.0, dtype=jnp.float32)

            def body(_, carry):
                model, key, sums = carry
                key, batch_key = jax.random.split(key)
                batch = planner_state.replay.sample(batch_key, planner_config.batch_size)
                model, metrics = update_option_model(
                    model, batch, planner_config.model_target_ema
                )
                sums = {name: sums[name] + metrics[name] for name in GRAPH_METRIC_NAMES}
                return model, key, sums

            model, _, sums = jax.lax.fori_loop(
                0, count, body,
                (planner_state.model, key, {name: zero for name in GRAPH_METRIC_NAMES}),
            )
            scale = jnp.maximum(count.astype(jnp.float32), 1.0)
            metrics = {name: value / scale for name, value in sums.items()}
            metrics["high_level_update_count"] = count
            return planner_state.replace(model=model), metrics

        def ppo_update(train_state, transitions, advantages, targets, initial_hidden, key):
            def epoch(carry, _):
                train_state, key = carry
                key, permutation_key = jax.random.split(key)
                permutation = jax.random.permutation(permutation_key, config.num_envs)
                envs_per_batch = config.num_envs // config.num_minibatches
                hidden_batches = initial_hidden[permutation].reshape(
                    (config.num_minibatches, envs_per_batch, config.hidden_size)
                )

                def split(x):
                    x = jnp.take(x, permutation, axis=1)
                    x = x.reshape(
                        (config.num_steps, config.num_minibatches, envs_per_batch)
                        + x.shape[2:]
                    )
                    return jnp.swapaxes(x, 0, 1)

                batches = jax.tree.map(split, (transitions, advantages, targets))

                def minibatch(train_state, batch):
                    initial_h, (trajectory, gae, target) = batch

                    def loss_fn(network_params):
                        _, logits, value = network.apply(
                            network_params, initial_h, trajectory.observation,
                            trajectory.rnn_reset, trajectory.objective,
                            trajectory.completed,
                        )
                        log_prob = categorical_log_prob(logits, trajectory.action)
                        ratio = jnp.exp(log_prob - trajectory.log_prob)
                        normalized = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_loss = -jnp.minimum(
                            ratio * normalized,
                            jnp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                            * normalized,
                        ).mean()
                        clipped = trajectory.value + jnp.clip(
                            value - trajectory.value, -config.clip_epsilon, config.clip_epsilon
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - target), jnp.square(clipped - target)
                        ).mean()
                        entropy = categorical_entropy(logits).mean()
                        loss = (
                            actor_loss
                            + config.value_coefficient * value_loss
                            - config.entropy_coefficient * entropy
                        )
                        return loss, (actor_loss, value_loss, entropy)

                    (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        train_state.params
                    )
                    norm = optax.global_norm(grads)
                    return train_state.apply_gradients(grads=grads), (loss, parts, norm)

                train_state, losses = jax.lax.scan(
                    minibatch, train_state, (hidden_batches, batches)
                )
                return (train_state, key), losses

            if frozen:
                zeros = jnp.zeros((config.update_epochs, config.num_minibatches))
                return train_state, key, (zeros, (zeros, zeros, zeros), zeros)
            (train_state, key), losses = jax.lax.scan(
                epoch, (train_state, key), None, length=config.update_epochs
            )
            return train_state, key, losses

        def update(runner_state, _):
            initial_hidden = runner_state[4]
            runner_state, (transitions, options) = jax.lax.scan(
                env_step, runner_state, None, length=config.num_steps
            )
            (train_state, env_state, observation, last_reset, hidden,
             goal_state, key, planner_state) = runner_state
            _, _, last_value = network.apply(
                train_state.params, hidden, observation[None], last_reset[None],
                goal_state.objective[None], goal_state.completed[None],
            )
            advantages, targets = calculate_gae(transitions, last_value[0], config)
            key, model_key = jax.random.split(key)
            planner_state, model_metrics = update_model(
                planner_state, options.recorded, model_key
            )
            train_state, key, losses = ppo_update(
                train_state, transitions, advantages, targets, initial_hidden, key
            )
            runner_state = (
                train_state, env_state, observation, last_reset, hidden,
                goal_state, key, planner_state,
            )

            ended = options.option_done
            ended_count = ended.astype(jnp.float32).sum()
            per_option = jnp.maximum(ended_count, 1.0)
            metrics = {
                "mean_reward": transitions.reward.mean(),
                "episode_end_rate": transitions.env_done.mean(),
                "goal_end_rate": (transitions.bootstrap_done & ~transitions.env_done).mean(),
                "loss": losses[0].mean(),
                "actor_loss": losses[1][0].mean(),
                "value_loss": losses[1][1].mean(),
                "entropy": losses[1][2].mean(),
                "gradient_norm": losses[2].mean(),
                "gradient_clip_fraction": (losses[2] > config.max_grad_norm).mean(),
                **model_metrics,
                "high_level_replay_size": planner_state.replay.size,
                "high_level_option_count": ended_count,
                "high_level_recorded_count": options.recorded.astype(jnp.float32).sum(),
                "selected_goal_distribution": (
                    jax.nn.one_hot(options.selected_goal, num_goals) * ended[..., None]
                ).sum((0, 1)) / per_option,
                "selected_goal_success_rate": options.success.astype(jnp.float32).sum() / per_option,
                "selected_new_goal_fraction": (
                    (options.selected_new & ended).astype(jnp.float32).sum() / per_option
                ),
                "planner_overflow_fraction": options.overflow.astype(jnp.float32).sum() / per_option,
                "planner_random_fraction": options.explored.astype(jnp.float32).sum() / per_option,
                "warmup_selection_fraction": (
                    (options.warmup & ended).astype(jnp.float32).sum() / per_option
                ),
                "mean_option_duration": (
                    (options.duration * ended).astype(jnp.float32).sum() / per_option
                ),
                "env_steps": planner_state.env_steps,
            }
            metrics.update(episode_metrics(transitions))

            update_count = train_state.step // (config.update_epochs * config.num_minibatches)
            if frozen:
                update_count = planner_state.env_steps // (config.num_envs * config.num_steps)
            if metrics_callback is not None:
                jax.debug.callback(metrics_callback, update_count, metrics, ordered=True)
            if progress_callback is not None:
                checkpoint = {
                    "policy": train_state.params,
                    "option_model": planner_state.model.train_state.params,
                    "option_model_target": planner_state.model.target_params,
                }

                def report(_):
                    jax.debug.callback(
                        progress_callback, update_count, metrics, checkpoint, ordered=True
                    )

                jax.lax.cond(
                    (update_count == 1)
                    | (update_count % progress_interval == 0)
                    | (update_count == num_updates),
                    report, lambda _: None, operand=None,
                )
            if return_rollout:
                return runner_state, (metrics, (transitions.objective, options))
            return runner_state, metrics

        runner_state, metrics = jax.lax.scan(update, runner_state, None, length=num_updates)
        if return_rollout:
            metrics, rollout = metrics
            return {"runner_state": runner_state, "metrics": metrics, "rollout": rollout}
        return {"runner_state": runner_state, "metrics": metrics}

    return train


__all__ = ["OptionStep", "PlannerTrainingState", "make_train"]
