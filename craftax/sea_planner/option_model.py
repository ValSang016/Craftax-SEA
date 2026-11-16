"""Option model F (state-conditioned achievement graph) and the K-hop planner.

For a situation ``x = (s_t, c_t, g_i)`` (symbolic observation, completed
achievements of the episode, current node) and a goal ``g_j``, F predicts

* ``p``   edge probability p(g_j succeeds | x);
* ``tau`` duration until success;
* ``d``   probability of dying during the option (trained, not used in J);
* ``c'``  completed vector after success;
* ``z'``  latent situation after success (input of the next hop);

and decodes inventory + intrinsics from latents.  Every head is trained by
supervised learning on observed option outcomes only.
"""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

# Classic symbolic observation (1345 values): 7x9x21 map (1323 values), then
# inventory (12) and intrinsics health/food/drink/energy (4), then direction
# (4), light level and the sleeping flag.
AUX_START = 1323
AUX_STOP = 1339
AUX_SIZE = AUX_STOP - AUX_START

LOSS_WEIGHTS = {
    "success": 1.0,
    "death": 1.0,
    "duration": 0.1,
    "completed": 1.0,
    "consistency": 1.0,
    "aux": 10.0,
}
GRAPH_METRIC_NAMES = (
    "graph_loss",
    *(f"graph_{name}_loss" for name in LOSS_WEIGHTS),
    "graph_brier",
    "graph_edge_state_std",
    "graph_mean_success_probability",
    "graph_batch_success_rate",
)


class EdgePrediction(NamedTuple):
    success_logit: jax.Array
    log_duration: jax.Array
    death_logit: jax.Array
    completed_logits: jax.Array
    next_latent: jax.Array


class OptionModel(nn.Module):
    num_goals: int = 22
    hidden_size: int = 256
    latent_size: int = 128
    goal_embedding_size: int = 32

    def setup(self):
        self.enc_1 = nn.Dense(2 * self.hidden_size)
        self.enc_2 = nn.Dense(self.hidden_size)
        self.enc_3 = nn.Dense(self.latent_size)
        self.enc_norm = nn.LayerNorm()
        self.goal_embed = nn.Embed(self.num_goals, self.goal_embedding_size)
        self.trans_1 = nn.Dense(self.hidden_size)
        self.trans_2 = nn.Dense(self.hidden_size)
        self.success = nn.Dense(1)
        self.duration = nn.Dense(1)
        self.death = nn.Dense(1)
        self.completed = nn.Dense(self.num_goals)
        self.latent_delta = nn.Dense(self.latent_size)
        self.next_norm = nn.LayerNorm()
        self.dec_1 = nn.Dense(self.hidden_size)
        self.dec_2 = nn.Dense(AUX_SIZE)

    def encode(self, observation, completed, node):
        # ``node == num_goals`` means "no current node" (start of an episode).
        features = jnp.concatenate(
            [
                observation.astype(jnp.float32),
                completed[..., : self.num_goals].astype(jnp.float32),
                jax.nn.one_hot(node, self.num_goals + 1),
            ],
            axis=-1,
        )
        hidden = nn.relu(self.enc_1(features))
        hidden = nn.relu(self.enc_2(hidden))
        return self.enc_norm(self.enc_3(hidden))

    def edge(self, latent, goal):
        hidden = jnp.concatenate([latent, self.goal_embed(goal)], axis=-1)
        hidden = nn.relu(self.trans_1(hidden))
        hidden = nn.relu(self.trans_2(hidden))
        return EdgePrediction(
            success_logit=self.success(hidden)[..., 0],
            log_duration=self.duration(hidden)[..., 0],
            death_logit=self.death(hidden)[..., 0],
            completed_logits=self.completed(hidden),
            next_latent=self.next_norm(latent + self.latent_delta(hidden)),
        )

    def decode(self, latent):
        return self.dec_2(nn.relu(self.dec_1(latent)))

    def __call__(self, observation, completed, node, goal):
        latent = self.encode(observation, completed, node)
        return self.edge(latent, goal), self.decode(latent)


@struct.dataclass
class OptionModelState:
    train_state: TrainState
    target_params: object


def create_option_model(key, observation, num_goals, config):
    model = OptionModel(
        num_goals=num_goals,
        hidden_size=config.model_hidden_size,
        latent_size=config.model_latent_size,
    )
    batch = observation.shape[0]
    params = model.init(
        key,
        observation,
        jnp.zeros((batch, num_goals), dtype=bool),
        jnp.full((batch,), num_goals, dtype=jnp.int32),
        jnp.zeros((batch,), dtype=jnp.int32),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(config.learning_rate)
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return OptionModelState(train_state, params)


def _encode(state_or_fn, params, observation, completed, node):
    return state_or_fn(params, observation, completed, node, method=OptionModel.encode)


def _edge(apply_fn, params, latent, goal):
    return apply_fn(params, latent, goal, method=OptionModel.edge)


def _decode(apply_fn, params, latent):
    return apply_fn(params, latent, method=OptionModel.decode)


def _success_mean(values, success):
    return (values * success).sum() / jnp.maximum(success.sum(), 1.0)


def option_model_loss(params, target_params, apply_fn, batch):
    """Supervised loss; post-success heads use successful options only."""
    success = batch.success.astype(jnp.float32)
    died = batch.died.astype(jnp.float32)
    latent = _encode(apply_fn, params, batch.observation, batch.completed, batch.current_node)
    prediction = _edge(apply_fn, params, latent, batch.goal)
    num_goals = prediction.completed_logits.shape[-1]

    success_loss = optax.sigmoid_binary_cross_entropy(prediction.success_logit, success).mean()
    death_loss = optax.sigmoid_binary_cross_entropy(prediction.death_logit, died).mean()
    log_duration = jnp.log(jnp.maximum(batch.duration, 1).astype(jnp.float32))
    duration_loss = _success_mean(jnp.square(prediction.log_duration - log_duration), success)
    completed_loss = _success_mean(
        optax.sigmoid_binary_cross_entropy(
            prediction.completed_logits,
            batch.next_completed[..., :num_goals].astype(jnp.float32),
        ).mean(-1),
        success,
    )
    target_latent = jax.lax.stop_gradient(
        _encode(
            apply_fn, target_params, batch.next_observation,
            batch.next_completed, batch.next_current_node,
        )
    )
    consistency_loss = _success_mean(
        jnp.square(prediction.next_latent - target_latent).mean(-1), success
    )
    aux_now = batch.observation[..., AUX_START:AUX_STOP]
    aux_next = batch.next_observation[..., AUX_START:AUX_STOP]
    aux_loss = jnp.square(_decode(apply_fn, params, latent) - aux_now).mean() + _success_mean(
        jnp.square(_decode(apply_fn, params, prediction.next_latent) - aux_next).mean(-1),
        success,
    )
    losses = {
        "success": success_loss,
        "death": death_loss,
        "duration": duration_loss,
        "completed": completed_loss,
        "consistency": consistency_loss,
        "aux": aux_loss,
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in losses.items())
    return total, (losses, jax.nn.sigmoid(prediction.success_logit), latent)


def update_option_model(state: OptionModelState, batch, target_ema):
    train_state = state.train_state
    (total, (losses, probability, latent)), grads = jax.value_and_grad(
        option_model_loss, has_aux=True
    )(train_state.params, state.target_params, train_state.apply_fn, batch)
    train_state = train_state.apply_gradients(grads=grads)
    target_params = optax.incremental_update(
        train_state.params, state.target_params, 1.0 - target_ema
    )

    # How much each edge depends on the state: std over the batch of p(g_j|x)
    # for every goal g_j, averaged over goals (zero for a static graph).
    num_goals = batch.completed.shape[-1]
    all_edges = _edge(
        train_state.apply_fn,
        train_state.params,
        jnp.repeat(jax.lax.stop_gradient(latent)[:, None, :], num_goals, axis=1),
        jnp.broadcast_to(jnp.arange(num_goals), latent.shape[:1] + (num_goals,)),
    )
    edge_probability = jax.nn.sigmoid(all_edges.success_logit)
    success = batch.success.astype(jnp.float32)
    metrics = {
        "graph_loss": total,
        **{f"graph_{name}_loss": value for name, value in losses.items()},
        "graph_brier": jnp.square(probability - success).mean(),
        "graph_edge_state_std": edge_probability.std(axis=0).mean(),
        "graph_mean_success_probability": probability.mean(),
        "graph_batch_success_rate": success.mean(),
    }
    return OptionModelState(train_state, target_params), metrics


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


def prerequisites_met(completed, graph):
    """Static readiness: all parents of a goal are completed.  [..., G]."""
    graph = graph.astype(bool)
    return jnp.all(~graph | completed[..., :, None], axis=-2)


def _all_edges(apply_fn, params, latent, num_goals, max_duration):
    """Predictions for every goal from ``latent [..., L]`` -> ``[..., G]``."""
    batch_shape = latent.shape[:-1] + (num_goals,)
    prediction = _edge(
        apply_fn,
        params,
        jnp.broadcast_to(latent[..., None, :], batch_shape + latent.shape[-1:]),
        jnp.broadcast_to(jnp.arange(num_goals), batch_shape),
    )
    probability = jax.nn.sigmoid(prediction.success_logit)
    duration = jnp.clip(jnp.exp(prediction.log_duration), 1.0, max_duration)
    completed_after = jax.nn.sigmoid(prediction.completed_logits) > 0.5
    return probability, duration, completed_after, prediction.next_latent


def plan_path_scores(
    state: OptionModelState,
    observation,
    completed,
    current_node,
    candidates,
    *,
    horizon,
    beam_width,
    gamma,
    use_leaf,
    max_duration,
    graph=None,
    prune="none",
    prune_width=13,
):
    """Best K-hop path score J for every first goal, shape ``[N, G]``.

    J = sum_k gamma^{T_k} (prod_{i<=k} p_i) 1[g_k new]
        (+ gamma^{T_K} (prod p) sum_j gamma^{tau_j} p_j 1[g_j new] 1[g_j candidate])

    Hop 1 enumerates every goal; each first goal then keeps a beam of
    ``beam_width`` partial paths ranked by partial score.  ``prune="leaf"``
    evaluates the leaf term only on (at most ``prune_width``) ready, new
    candidate goals; ``"all"`` also forbids unready goals at hops >= 2.
    """
    if prune not in ("none", "leaf", "all"):
        raise ValueError("prune must be none, leaf, or all")
    if prune != "none" and graph is None:
        raise ValueError("prerequisite pruning needs the static graph")
    apply_fn = state.train_state.apply_fn
    params = state.train_state.params
    num_goals = candidates.shape[-1]
    identity = jnp.eye(num_goals, dtype=bool)
    done_before = completed[..., :num_goals].astype(bool)

    # Hop 1, axes [N, G1].
    latent = _encode(apply_fn, params, observation, completed, current_node)
    probability, duration, completed_after, next_latent = _all_edges(
        apply_fn, params, latent, num_goals, max_duration
    )
    path_probability = probability
    elapsed = duration
    score = gamma**elapsed * path_probability * (~done_before)
    first_hop_score = score
    path_completed = done_before[..., None, :] | identity | completed_after

    # Beam axis: [N, G1, B].
    path_probability, elapsed, score = (x[..., None] for x in (path_probability, elapsed, score))
    path_completed = path_completed[..., None, :]
    latent = next_latent[..., None, :]

    for _ in range(horizon - 1):
        probability, duration, completed_after, next_latent = _all_edges(
            apply_fn, params, latent, num_goals, max_duration
        )  # [N, G1, B, G]
        step_probability = path_probability[..., None] * probability
        step_elapsed = elapsed[..., None] + duration
        step_score = score[..., None] + gamma**step_elapsed * step_probability * (~path_completed)
        allowed = jnp.broadcast_to(candidates[..., None, None, :], step_score.shape)
        if prune == "all":
            allowed = allowed & prerequisites_met(path_completed, graph)
        step_score = jnp.where(allowed, step_score, -jnp.inf)
        step_completed = path_completed[..., None, :] | identity | completed_after

        options = step_score.shape[-2] * num_goals
        width = min(beam_width, options)

        def flatten(x):
            return x.reshape(x.shape[:2] + (options,) + x.shape[4:])

        _, keep = jax.lax.top_k(flatten(step_score), width)  # [N, G1, width]

        def take(x):
            x = flatten(x)
            index = keep.reshape(keep.shape + (1,) * (x.ndim - 3))
            return jnp.take_along_axis(x, index, axis=2)

        path_probability = take(step_probability)
        elapsed = take(step_elapsed)
        score = take(step_score)
        path_completed = take(step_completed)
        latent = take(next_latent)

    if use_leaf:
        new_candidates = ~path_completed & candidates[..., None, None, :]
        if prune == "none":
            probability, duration, _, _ = _all_edges(
                apply_fn, params, latent, num_goals, max_duration
            )
            leaf = (gamma**duration * probability * new_candidates).sum(-1)
        else:
            eligible = new_candidates & prerequisites_met(path_completed, graph)
            width = min(prune_width, num_goals)
            # Eligible goals first, lower index first among ties.
            priority = eligible - 1e-3 * jnp.arange(num_goals) / num_goals
            _, goals = jax.lax.top_k(priority, width)
            valid = jnp.take_along_axis(eligible, goals, axis=-1)
            prediction = _edge(
                apply_fn,
                params,
                jnp.broadcast_to(latent[..., None, :], goals.shape + latent.shape[-1:]),
                goals,
            )
            probability = jax.nn.sigmoid(prediction.success_logit)
            duration = jnp.clip(jnp.exp(prediction.log_duration), 1.0, max_duration)
            leaf = (gamma**duration * probability * valid).sum(-1)
        score = score + gamma**elapsed * path_probability * leaf

    best = score.max(axis=-1)
    # A first goal whose every continuation is forbidden keeps its 1-hop value.
    return jnp.where(jnp.isfinite(best), best, first_hop_score)


def sample_from_scores(key, scores, candidates, temperature):
    """Softmax over J divided by the best candidate's J (scale-free)."""
    best = jnp.max(jnp.where(candidates, scores, -jnp.inf), axis=-1, keepdims=True)
    logits = scores / jnp.maximum(best, 1e-6) / temperature
    logits = jnp.where(candidates, logits, -jnp.inf)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def choose_planner_goals(
    key, state, observation, completed, current_node, candidates, needs_goal, config,
    graph=None,
):
    """K-hop planning for at most ``planner_max_decisions`` envs per step.

    Envs needing a goal beyond the budget get a 1-hop plan, computed only when
    the budget actually overflows.  Returns ``(goals, overflow_mask)``.
    """
    num_envs = observation.shape[0]
    budget = min(config.planner_max_decisions, num_envs)
    priority_key, plan_key, fallback_key = jax.random.split(key, 3)
    plan = dict(
        beam_width=config.planner_beam_width,
        gamma=config.planner_gamma,
        use_leaf=config.planner_leaf,
        max_duration=float(config.goal_timeout),
        graph=graph,
        prune=config.planner_prune,
        prune_width=config.planner_prune_width,
    )
    if config.planner_prune == "all":
        # Hop 1 is pruned through the candidates, if any ready goal exists.
        ready = prerequisites_met(completed[..., : candidates.shape[-1]], graph) & candidates
        candidates = jnp.where(ready.any(-1, keepdims=True), ready, candidates)

    def one_hop(_):
        scores = plan_path_scores(
            state, observation, completed, current_node, candidates, horizon=1, **plan
        )
        return sample_from_scores(fallback_key, scores, candidates, config.planner_temperature)

    if config.planner_horizon == 1 and budget == num_envs:
        return one_hop(None), jnp.zeros((num_envs,), dtype=bool)

    # Envs that need a goal first, in random order.
    priority = needs_goal + 0.5 * jax.random.uniform(priority_key, (num_envs,))
    _, chosen = jax.lax.top_k(priority, budget)
    scores = plan_path_scores(
        state,
        observation[chosen],
        completed[chosen],
        current_node[chosen],
        candidates[chosen],
        horizon=config.planner_horizon,
        **plan,
    )
    planned = sample_from_scores(plan_key, scores, candidates[chosen], config.planner_temperature)
    was_planned = jnp.zeros((num_envs,), dtype=bool).at[chosen].set(True)
    overflow = needs_goal & ~was_planned
    goals = jax.lax.cond(
        overflow.any(),
        one_hop,
        lambda _: jnp.zeros((num_envs,), dtype=jnp.int32),
        operand=None,
    )
    return goals.at[chosen].set(planned), overflow


__all__ = [
    "AUX_SIZE",
    "AUX_START",
    "AUX_STOP",
    "EdgePrediction",
    "GRAPH_METRIC_NAMES",
    "LOSS_WEIGHTS",
    "OptionModel",
    "OptionModelState",
    "choose_planner_goals",
    "create_option_model",
    "option_model_loss",
    "plan_path_scores",
    "prerequisites_met",
    "sample_from_scores",
    "update_option_model",
]
