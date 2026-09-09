"""SEA transition representation learning, clustering and graph recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from sklearn.cluster import KMeans

from craftax.sea.networks import TransitionEncoder


@dataclass
class TransitionDataset:
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray
    event_count: np.ndarray
    episode_id: np.ndarray
    episode_step: np.ndarray

    def __len__(self):
        return int(self.action.shape[0])

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(**{name: data[name] for name in cls.__annotations__})


@dataclass(frozen=True)
class DiscoveryConfig:
    learning_rate: float = 2e-4
    batch_size: int = 256
    train_steps: int = 10_000
    hidden_size: int = 256
    embedding_size: int = 128
    contrast_weight: float = 20.0
    contrast_groups: int = 32
    max_events_per_episode: int = 8
    max_grad_norm: float = 1.0


@dataclass
class ClusterArtifacts:
    centroids: np.ndarray
    threshold: float
    graph: np.ndarray
    labels: np.ndarray

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            centroids=self.centroids,
            threshold=np.asarray(self.threshold, dtype=np.float32),
            graph=self.graph,
            labels=self.labels,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(
                centroids=data["centroids"],
                threshold=float(data["threshold"]),
                graph=data["graph"],
                labels=data["labels"],
            )


def determinant_contrast_loss(embeddings, valid):
    """Negative determinant used by SEA to separate events in one episode."""

    distances = jnp.square(embeddings[:, :, None, :] - embeddings[:, None, :, :]).sum(
        -1
    )
    valid_pair = valid[:, :, None] & valid[:, None, :]
    scale = jnp.max(jnp.where(valid_pair, distances, 0.0), axis=(1, 2))
    scale = jnp.maximum(scale, 1e-6)
    kernel = jnp.exp(-2.0 * distances / scale[:, None, None])
    kernel = jnp.where(valid_pair, kernel, 0.0)
    # Padding contributes an identity block, leaving the determinant of the
    # valid submatrix unchanged.
    invalid_diagonal = jnp.eye(embeddings.shape[1], dtype=embeddings.dtype)[None]
    invalid_diagonal *= (~valid)[:, :, None]
    kernel = kernel + invalid_diagonal
    determinants = jnp.linalg.det(kernel)
    usable = valid.sum(axis=1) >= 2
    denominator = jnp.maximum(usable.sum(), 1)
    return -(determinants * usable).sum() / denominator


def _contrast_indices(dataset, rng, config):
    positive = np.flatnonzero(dataset.event_count > 0)
    by_episode = {}
    for index in positive:
        by_episode.setdefault(int(dataset.episode_id[index]), []).append(index)
    usable = [np.asarray(values) for values in by_episode.values() if len(values) >= 2]
    if not usable:
        shape = (config.contrast_groups, config.max_events_per_episode)
        return np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=bool)

    groups = rng.choice(len(usable), size=config.contrast_groups, replace=True)
    indices = np.zeros(
        (config.contrast_groups, config.max_events_per_episode), dtype=np.int64
    )
    valid = np.zeros_like(indices, dtype=bool)
    for row, group in enumerate(groups):
        candidates = usable[group]
        count = min(len(candidates), config.max_events_per_episode)
        chosen = rng.choice(candidates, size=count, replace=False)
        indices[row, :count] = chosen
        valid[row, :count] = True
    return indices, valid


def train_transition_encoder(
    dataset: TransitionDataset,
    config: DiscoveryConfig = DiscoveryConfig(),
    *,
    pixel: bool = False,
    seed: int = 0,
):
    if len(dataset) == 0:
        raise ValueError("cannot train discovery model on an empty dataset")
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    network = TransitionEncoder(
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
        pixel=pixel,
    )
    sample = slice(0, 1)
    params = network.init(
        key,
        jnp.asarray(dataset.observation[sample]),
        jnp.asarray(dataset.action[sample]),
        jnp.asarray(dataset.next_observation[sample]),
        jnp.asarray(dataset.terminal[sample]),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    @jax.jit
    def update(state, batch, contrast_batch, valid):
        def loss_fn(parameters):
            reward_logit, _ = network.apply(parameters, *batch[:4])
            target = (batch[4] > 0).astype(jnp.float32)
            prediction = jax.nn.sigmoid(reward_logit)
            prediction_loss = jnp.square(prediction - target).mean()
            _, contrast_embeddings = network.apply(parameters, *contrast_batch)
            contrast_loss = determinant_contrast_loss(contrast_embeddings, valid)
            total = prediction_loss + config.contrast_weight * contrast_loss
            return total, (prediction_loss, contrast_loss)

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        return state.apply_gradients(grads=gradients), (loss, metrics)

    metrics = None
    for _ in range(config.train_steps):
        batch_indices = rng.integers(0, len(dataset), size=config.batch_size)
        group_indices, valid = _contrast_indices(dataset, rng, config)
        batch = (
            jnp.asarray(dataset.observation[batch_indices]),
            jnp.asarray(dataset.action[batch_indices]),
            jnp.asarray(dataset.next_observation[batch_indices]),
            jnp.asarray(dataset.terminal[batch_indices]),
            jnp.asarray(dataset.event_count[batch_indices]),
        )
        contrast_batch = (
            jnp.asarray(dataset.observation[group_indices]),
            jnp.asarray(dataset.action[group_indices]),
            jnp.asarray(dataset.next_observation[group_indices]),
            jnp.asarray(dataset.terminal[group_indices]),
        )
        state, metrics = update(state, batch, contrast_batch, jnp.asarray(valid))
    return network, state, metrics


def embed_positive_transitions(network, params, dataset, batch_size=1024):
    indices = np.flatnonzero(dataset.event_count > 0)
    chunks = []

    @jax.jit
    def embed(observation, action, next_observation, terminal):
        return network.apply(params, observation, action, next_observation, terminal)[1]

    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        chunks.append(
            np.asarray(
                embed(
                    jnp.asarray(dataset.observation[selected]),
                    jnp.asarray(dataset.action[selected]),
                    jnp.asarray(dataset.next_observation[selected]),
                    jnp.asarray(dataset.terminal[selected]),
                )
            )
        )
    if not chunks:
        raise ValueError("dataset contains no positive achievement transitions")
    return (
        np.concatenate(chunks),
        dataset.episode_id[indices],
        dataset.episode_step[indices],
    )


def _count_conflicts(labels, episode_id):
    conflicts = 0
    for episode in np.unique(episode_id):
        values = labels[episode_id == episode]
        _, counts = np.unique(values, return_counts=True)
        conflicts += int(((counts * (counts - 1)) // 2).sum())
    return conflicts


def _transitive_reduction(graph):
    graph = graph.astype(bool).copy()
    count = graph.shape[0]
    for source in range(count):
        for target in range(count):
            if not graph[source, target]:
                continue
            graph[source, target] = False
            reachable = np.zeros(count, dtype=bool)
            frontier = [source]
            while frontier:
                node = frontier.pop()
                for child in np.flatnonzero(graph[node]):
                    if not reachable[child]:
                        reachable[child] = True
                        frontier.append(int(child))
            if not reachable[target]:
                graph[source, target] = True
    return graph.astype(np.int8)


def fit_clusters(
    embeddings,
    episode_id,
    episode_step,
    *,
    min_clusters=None,
    max_clusters=30,
    conflict_fraction=0.001,
    random_state=0,
):
    """Fit SEA's constrained KMeans and recover its temporal graph."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    episode_id = np.asarray(episode_id)
    episode_step = np.asarray(episode_step)
    if len(embeddings) < 2:
        raise ValueError("at least two positive transitions are required")

    if min_clusters is None:
        lengths = sorted(
            [int((episode_id == ep).sum()) for ep in np.unique(episode_id)],
            reverse=True,
        )
        min_clusters = lengths[5] if len(lengths) >= 6 else max(lengths)
    min_clusters = int(np.clip(min_clusters, 2, len(embeddings)))
    max_clusters = int(np.clip(max_clusters, min_clusters, len(embeddings)))

    fitted = None
    labels = None
    for cluster_count in range(min_clusters, max_clusters + 1):
        fitted = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
        labels = fitted.fit_predict(embeddings)
        if _count_conflicts(labels, episode_id) < len(embeddings) * conflict_fraction:
            break

    centroids = fitted.cluster_centers_.astype(np.float32)
    distances = np.square(embeddings[:, None] - centroids[None]).sum(-1)
    assigned = distances[np.arange(len(labels)), labels]
    in_thresholds = []
    out_thresholds = []
    for cluster in range(len(centroids)):
        own = assigned[labels == cluster]
        other = distances[labels != cluster, cluster]
        if len(own):
            in_thresholds.append(float(np.quantile(own, 0.99)))
        if len(other):
            out_thresholds.append(float(np.quantile(other, 0.01)))
    in_edge = max(in_thresholds, default=float(assigned.max()))
    out_edge = min(out_thresholds, default=in_edge * 1.1 + 1e-6)
    threshold = float(max(1e-8, (in_edge + out_edge) / 2.0))

    order = np.zeros((len(centroids), len(centroids)), dtype=np.int64)
    happens = np.zeros((len(centroids),), dtype=np.int64)
    for episode in np.unique(episode_id):
        indexes = np.flatnonzero(episode_id == episode)
        indexes = indexes[np.argsort(episode_step[indexes])]
        sequence = labels[indexes]
        for position, source in enumerate(sequence):
            happens[source] += 1
            for target in sequence[position + 1 :]:
                order[source, target] += 1

    graph = np.zeros_like(order, dtype=np.int8)
    for source in range(len(centroids)):
        for target in range(len(centroids)):
            forward = order[source, target]
            if forward == 0:
                continue
            usually_before = forward / max(1, happens[target]) > 0.97
            almost_never_reverse = order[target, source] / forward < 0.001
            graph[source, target] = usually_before & almost_never_reverse
    graph = _transitive_reduction(graph)
    return ClusterArtifacts(centroids, threshold, graph, labels.astype(np.int32))


__all__ = [
    "ClusterArtifacts",
    "DiscoveryConfig",
    "TransitionDataset",
    "determinant_contrast_loss",
    "embed_positive_transitions",
    "fit_clusters",
    "train_transition_encoder",
]
