"""State-conditioned achievement graph learned by supervision, plus a K-hop planner.

The graph is an option model ``F``.  For the current situation
``x = (s_t, c_t, g_i)`` and any candidate goal ``g_j`` it predicts

* ``p``   – probability that the worker achieves ``g_j`` before the option ends
            (this is the state-conditioned edge weight ``g_i -> g_j``);
* ``tau`` – option duration given success;
* ``d``   – probability that the player dies during the option;
* ``c'``  – completed-achievement vector after success;
* ``z'``  – latent situation after success (fed to the next hop);
* inventory + intrinsics (health/food/drink/energy) decoded from latents.

Every head is fitted by supervised learning on observed option outcomes.  No
returns, bootstrapping or policy gradients are used for the graph.

The planner rolls ``F`` for K hops and scores a path by

    J = sum_k gamma^{T_k} * prod_{i<=k} p_i * 1[g_k not completed]
        (+ gamma^{T_K} * prod p * sum_j gamma^{tau_j} p_j 1[g_j not completed])

The optional last term is the discounted out-degree of the end node in the
predicted graph (``planner_leaf``).  The first goal is sampled from a softmax
over the best path score per first goal.
"""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

# Classic symbolic observation: 7x9x21 map, then 12 inventory and 4 intrinsics
# (health, food, drink, energy).  All are divided by 10 by the renderer.
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


class OptionModelOutput(NamedTuple):
    success_logit: jax.Array
    log_duration: jax.Array
    death_logit: jax.Array
    completed_logits: jax.Array
    next_latent: jax.Array


class OptionModel(nn.Module):
    num_goals: int
    hidden_size: int = 256
    latent_size: int = 128
    goal_embedding_size: int = 32

    def setup(self):
        self.encoder_layers = [
            nn.Dense(2 * self.hidden_size),
            nn.Dense(self.hidden_size),
            nn.Dense(self.latent_size),
        ]
        self.encoder_norm = nn.LayerNorm()
        self.goal_embedding = nn.Embed(self.num_goals, self.goal_embedding_size)
        self.transition_layers = [
            nn.Dense(self.hidden_size),
            nn.Dense(self.hidden_size),
        ]
        self.success_head = nn.Dense(1)
        self.duration_head = nn.Dense(1)
        self.death_head = nn.Dense(1)
        self.completed_head = nn.Dense(self.num_goals)
        self.delta_head = nn.Dense(self.latent_size)
        self.next_norm = nn.LayerNorm()
        self.decoder_layers = [nn.Dense(self.hidden_size), nn.Dense(AUX_SIZE)]

    def encode(self, observation, completed, node):
        x = jnp.concatenate(
            [
                observation.astype(jnp.float32),
                completed.astype(jnp.float32),
                jax.nn.one_hot(node, self.num_goals + 1),
            ],
            axis=-1,
        )
        x = nn.relu(self.encoder_layers[0](x))
        x = nn.relu(self.encoder_layers[1](x))
        return self.encoder_norm(self.encoder_layers[2](x))

    def transition(self, latent, goal):
        x = jnp.concatenate([latent, self.goal_embedding(goal)], axis=-1)
        for layer in self.transition_layers:
            x = nn.relu(layer(x))
        return OptionModelOutput(
            success_logit=self.success_head(x)[..., 0],
            log_duration=self.duration_head(x)[..., 0],
            death_logit=self.death_head(x)[..., 0],
            completed_logits=self.completed_head(x),
            next_latent=self.next_norm(latent + self.delta_head(x)),
        )

    def decode(self, latent):
        return self.decoder_layers[1](nn.relu(self.decoder_layers[0](latent)))

    def __call__(self, observation, completed, node, goal):
        latent = self.encode(observation, completed, node)
        return self.transition(latent, goal), self.decode(latent)


@struct.dataclass
class GraphPlannerState:
    train_state: TrainState
    target_params: object


def initialize_graph_planner(
    key, observation, completed, current_node, num_goals, config
):
    model = OptionModel(
        num_goals=num_goals,
        hidden_size=config.model_hidden_size,
        latent_size=config.model_latent_size,
    )
    goal = jnp.zeros(current_node.shape, dtype=jnp.int32)
    params = model.init(key, observation, completed, current_node, goal)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(config.learning_rate)
        ),
    )
    return GraphPlannerState(train_state, params)


def _encode(apply_fn, params, observation, completed, node):
    return apply_fn(params, observation, completed, node, method=OptionModel.encode)


def _transition(apply_fn, params, latent, goal):
    return apply_fn(params, latent, goal, method=OptionModel.transition)


def _decode(apply_fn, params, latent):
    return apply_fn(params, latent, method=OptionModel.decode)


def _masked_mean(values, weights):
    return (values * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def graph_planner_loss(params, target_params, apply_fn, batch):
    """Supervised option-model loss on replayed high-level transitions.

    Post-success heads (duration, completed, next latent) are fitted only on
    successful options because the planner conditions each hop on success.
    """

    latent = _encode(apply_fn, params, batch.observation, batch.completed, batch.current_node)
    output = _transition(apply_fn, params, latent, batch.goal)
    success = batch.success.astype(jnp.float32)
    died = jnp.asarray(batch.died).astype(jnp.float32)

    success_loss = optax.sigmoid_binary_cross_entropy(
        output.success_logit, success
    ).mean()
    death_loss = optax.sigmoid_binary_cross_entropy(output.death_logit, died).mean()
    duration_target = jnp.log(jnp.maximum(batch.duration, 1).astype(jnp.float32))
    duration_loss = _masked_mean(
        jnp.square(output.log_duration - duration_target), success
    )
    completed_loss = _masked_mean(
        optax.sigmoid_binary_cross_entropy(
            output.completed_logits,
            batch.next_completed[..., : output.completed_logits.shape[-1]].astype(
                jnp.float32
            ),
        ).mean(-1),
        success,
    )
    target_latent = jax.lax.stop_gradient(
        _encode(
            apply_fn,
            target_params,
            batch.next_observation,
            batch.next_completed,
            batch.next_current_node,
        )
    )
    consistency_loss = _masked_mean(
        jnp.square(output.next_latent - target_latent).mean(-1), success
    )
    current_aux = batch.observation[..., AUX_START:AUX_STOP]
    next_aux = batch.next_observation[..., AUX_START:AUX_STOP]
    aux_loss = jnp.square(_decode(apply_fn, params, latent) - current_aux).mean() + _masked_mean(
        jnp.square(_decode(apply_fn, params, output.next_latent) - next_aux).mean(-1),
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
    probability = jax.nn.sigmoid(output.success_logit)
    return total, (losses, probability, latent)


def update_graph_planner(state: GraphPlannerState, batch, target_ema, num_goals):
    train_state = state.train_state
    (total, (losses, probability, latent)), gradients = jax.value_and_grad(
        graph_planner_loss, has_aux=True
    )(train_state.params, state.target_params, train_state.apply_fn, batch)
    train_state = train_state.apply_gradients(grads=gradients)
    target_params = jax.tree.map(
        lambda target, online: target_ema * target + (1.0 - target_ema) * online,
        state.target_params,
        train_state.params,
    )

    # How much each edge varies with state: std over the batch of p(g_j | x)
    # for every goal, averaged over goals.  A static graph gives zero.
    all_goals = jnp.arange(num_goals)
    edge_output = _transition(
        train_state.apply_fn,
        jax.lax.stop_gradient(train_state.params),
        jnp.broadcast_to(
            jax.lax.stop_gradient(latent)[:, None, :],
            latent.shape[:1] + (num_goals, latent.shape[-1]),
        ),
        jnp.broadcast_to(all_goals, latent.shape[:1] + (num_goals,)),
    )
    edge_probability = jax.nn.sigmoid(edge_output.success_logit)
    success = batch.success.astype(jnp.float32)
    metrics = {
        "graph_loss": total,
        **{f"graph_{name}_loss": value for name, value in losses.items()},
        "graph_batch_success_rate": success.mean(),
        "graph_mean_success_probability": probability.mean(),
        "graph_brier": jnp.square(probability - success).mean(),
        "graph_edge_state_std": edge_probability.std(axis=0).mean(),
    }
    return GraphPlannerState(train_state, target_params), metrics


GRAPH_METRIC_NAMES = (
    "graph_loss",
    *(f"graph_{name}_loss" for name in LOSS_WEIGHTS),
    "graph_batch_success_rate",
    "graph_mean_success_probability",
    "graph_brier",
    "graph_edge_state_std",
)


def _expand(apply_fn, params, latent, num_goals, max_duration):
    """Evaluate every goal from ``latent[..., L]`` -> outputs ``[..., G]``."""
    shape = latent.shape[:-1] + (num_goals,)
    output = _transition(
        apply_fn,
        params,
        jnp.broadcast_to(latent[..., None, :], shape + latent.shape[-1:]),
        jnp.broadcast_to(jnp.arange(num_goals), shape),
    )
    probability = jax.nn.sigmoid(output.success_logit)
    duration = jnp.clip(jnp.exp(output.log_duration), 1.0, max_duration)
    completed_probability = jax.nn.sigmoid(output.completed_logits)
    return probability, duration, completed_probability, output.next_latent


def prerequisites_met(completed, graph):
    """Static-graph readiness: every parent of a node is completed. [..., G]."""
    num_goals = completed.shape[-1]
    graph = graph.astype(bool)
    if graph.shape[0] < num_goals:  # e.g. an extra "unknown" objective
        pad = num_goals - graph.shape[0]
        graph = jnp.pad(graph, ((0, pad), (0, pad)))
    return jnp.all(~graph | completed[..., :, None], axis=-2)


def plan_path_scores(
    state: GraphPlannerState,
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
    """Return the best K-hop path score for every first goal, shape [N, G].

    Hop 1 enumerates all goals.  Each first goal then keeps its own beam of
    ``beam_width`` partial paths ranked by partial score.

    ``prune`` uses the static prerequisite graph on the predicted completed
    vector: "leaf" evaluates the end-node out-degree only on (at most
    ``prune_width``) ready, uncompleted goals; "all" additionally forbids
    goals with unmet prerequisites at every later hop (completed goals stay
    allowed).  Hop 1 pruning is applied by the caller through ``candidates``.
    """
    if prune not in ("none", "leaf", "all"):
        raise ValueError("prune must be none, leaf, or all")
    if prune != "none" and graph is None:
        raise ValueError("prerequisite pruning needs the static graph")

    params = state.train_state.params
    apply_fn = state.train_state.apply_fn
    num_goals = candidates.shape[-1]
    goal_completed = completed[..., :num_goals].astype(bool)
    eye = jnp.eye(num_goals, dtype=bool)

    latent = _encode(apply_fn, params, observation, completed, current_node)
    probability, duration, completed_probability, next_latent = _expand(
        apply_fn, params, latent, num_goals, max_duration
    )
    new = (~goal_completed).astype(jnp.float32)
    path_probability = probability
    elapsed = duration
    score = (gamma**elapsed) * path_probability * new
    path_completed = (
        goal_completed[..., None, :] | eye | (completed_probability > 0.5)
    )
    # Beam axis: [N, G_first, B, ...] starting with one beam per first goal.
    path_probability = path_probability[..., None]
    elapsed = elapsed[..., None]
    score = score[..., None]
    path_completed = path_completed[..., None, :]
    latent = next_latent[..., None, :]

    for _ in range(horizon - 1):
        probability, duration, completed_probability, next_latent = _expand(
            apply_fn, params, latent, num_goals, max_duration
        )  # [N, G1, B, G2]
        new = (~path_completed).astype(jnp.float32)
        step_probability = path_probability[..., None] * probability
        step_elapsed = elapsed[..., None] + duration
        step_score = score[..., None] + (gamma**step_elapsed) * step_probability * new
        allowed = candidates[..., None, None, :]
        if prune == "all":
            allowed = allowed & prerequisites_met(path_completed, graph)
        step_score = jnp.where(allowed, step_score, -jnp.inf)
        ranking = step_score
        step_completed = (
            path_completed[..., None, :] | eye | (completed_probability > 0.5)
        )  # [N, G1, B, G2, G]
        beams = ranking.shape[-2] * num_goals
        width = min(beam_width, beams)
        flat = lambda x: x.reshape(x.shape[:2] + (beams,) + x.shape[4:])
        _, keep = jax.lax.top_k(flat(ranking), width)  # [N, G1, width]

        def gather(x):
            x = flat(x)
            index = keep.reshape(keep.shape + (1,) * (x.ndim - 3))
            return jnp.take_along_axis(x, index, axis=2)

        path_probability = gather(step_probability)
        elapsed = gather(step_elapsed)
        score = gather(step_score)
        path_completed = gather(step_completed)
        latent = gather(next_latent)

    if use_leaf and prune != "none":
        ready = (
            prerequisites_met(path_completed, graph)
            & ~path_completed
            & candidates[..., None, None, :]
        )  # [N, G1, B, G]
        width = min(prune_width, num_goals)
        # Ready goals first, lowest index first among ties.
        priority = ready.astype(jnp.float32) - 1e-3 * jnp.arange(num_goals) / num_goals
        _, goals = jax.lax.top_k(priority, width)
        valid = jnp.take_along_axis(ready, goals, axis=-1)
        output = _transition(
            apply_fn,
            params,
            jnp.broadcast_to(latent[..., None, :], goals.shape + latent.shape[-1:]),
            goals,
        )
        probability = jax.nn.sigmoid(output.success_logit)
        duration = jnp.clip(jnp.exp(output.log_duration), 1.0, max_duration)
        out_degree = ((gamma**duration) * probability * valid).sum(-1)
        score = score + (gamma**elapsed) * path_probability * out_degree
    elif use_leaf:
        probability, duration, _, _ = _expand(
            apply_fn, params, latent, num_goals, max_duration
        )
        out_degree = (
            (gamma**duration)
            * probability
            * (~path_completed).astype(jnp.float32)
            * candidates[..., None, None, :]
        ).sum(-1)
        score = score + (gamma**elapsed) * path_probability * out_degree

    return score.max(axis=-1)


def sample_from_scores(key, scores, candidates, temperature):
    """Softmax over path scores divided by the best candidate's score.

    J is an expected count of new achievements and is often far below 1, so a
    fixed temperature on raw J is nearly uniform.  Normalising by the per-
    decision maximum makes ``temperature`` the logit gap (in units of 1/T)
    between the best path and a path worth nothing (e.g. a pure repeat).
    """
    best = jnp.max(jnp.where(candidates, scores, -jnp.inf), axis=-1, keepdims=True)
    scale = jnp.maximum(best, 1e-6)
    logits = jnp.where(candidates, scores / scale / temperature, -jnp.inf)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def choose_planner_goals(
    key, state, observation, completed, current_node, candidates, needs_goal, config,
    graph=None,
):
    """Plan for at most ``planner_max_decisions`` environments per step.

    Environments that need a goal beyond that budget fall back to a 1-hop
    plan, which is cheap for the whole batch.  The returned overflow mask
    lets training report how often that happens.
    """

    num_envs = observation.shape[0]
    limit = min(config.planner_max_decisions, num_envs)
    priority_key, plan_key, fallback_key = jax.random.split(key, 3)
    prune = config.planner_prune
    common = dict(
        beam_width=config.planner_beam_width,
        gamma=config.planner_gamma,
        use_leaf=config.planner_leaf,
        max_duration=float(config.goal_timeout),
        graph=graph,
        prune=prune,
        prune_width=config.planner_prune_width,
    )
    if prune == "all":
        ready = prerequisites_met(completed, graph) & candidates
        candidates = jnp.where(ready.any(-1, keepdims=True), ready, candidates)

    def one_hop(_):
        scores = plan_path_scores(
            state, observation, completed, current_node, candidates,
            horizon=1, **common,
        )
        return sample_from_scores(
            fallback_key, scores, candidates, config.planner_temperature
        )

    if config.planner_horizon == 1 and limit == num_envs:
        return one_hop(None), jnp.zeros((num_envs,), dtype=bool)

    priority = needs_goal.astype(jnp.float32) + 0.5 * jax.random.uniform(
        priority_key, (num_envs,)
    )
    _, index = jax.lax.top_k(priority, limit)
    planned_scores = plan_path_scores(
        state,
        observation[index],
        completed[index],
        current_node[index],
        candidates[index],
        horizon=config.planner_horizon,
        **common,
    )
    planned = sample_from_scores(
        plan_key, planned_scores, candidates[index], config.planner_temperature
    )
    planned_mask = jnp.zeros((num_envs,), dtype=bool).at[index].set(True)
    overflow = needs_goal & ~planned_mask
    # The whole-batch 1-hop fallback is only computed when the K-hop budget
    # actually overflowed in this step.
    selected = jax.lax.cond(
        overflow.any(), one_hop,
        lambda _: jnp.zeros((num_envs,), dtype=jnp.int32), operand=None,
    )
    selected = selected.at[index].set(planned)
    return selected, overflow


__all__ = [
    "AUX_SIZE",
    "AUX_START",
    "AUX_STOP",
    "GRAPH_METRIC_NAMES",
    "GraphPlannerState",
    "OptionModel",
    "choose_planner_goals",
    "graph_planner_loss",
    "initialize_graph_planner",
    "plan_path_scores",
    "prerequisites_met",
    "sample_from_scores",
    "update_graph_planner",
]
