"""SEA transition representation learning, clustering and graph recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from craftax.sea.networks import TransitionEncoder
from craftax.sea.graph import infer_graph, transitive_reduction as _transitive_reduction


@dataclass
class TransitionDataset:
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray
    event_count: np.ndarray
    episode_id: np.ndarray
    episode_step: np.ndarray
    contrast_eligible: np.ndarray
    new_achievements: np.ndarray | None = None

    def __len__(self):
        return int(self.action.shape[0])

    def save(self, path):
        np.savez_compressed(
            path,
            **{key: value for key, value in self.__dict__.items() if value is not None},
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            values = {name: data[name] for name in data.files}
            # Datasets written by the initial port predate the 1M-step
            # contrast window marker.  Treat them as entirely eligible.
            values.setdefault(
                "contrast_eligible",
                np.ones_like(values["event_count"], dtype=bool),
            )
            values.setdefault("new_achievements", None)
            return cls(**{name: values[name] for name in cls.__annotations__})


@dataclass(frozen=True)
class DiscoveryConfig:
    learning_rate: float = 1e-4
    batch_size: int = 2560
    train_steps: int = 19_532
    hidden_size: int = 256  # Original SEA transition-state width.
    embedding_size: int = 256
    contrast_weight: float = 20.0
    contrast_groups: int = 128
    contrast_episode_capacity: int = 256
    max_events_per_episode: int = 8
    max_grad_norm: float = 1.0

    @property
    def mean_contrast_coefficient(self):
        # Normalize the reference SEA objective (2560 prediction terms,
        # 128 contrast groups, summed contrast multiplier 20). Changing the
        # working batch/group count must not change this relative weight.
        return self.contrast_weight * 128 / 2560


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
    positive = np.flatnonzero((dataset.event_count > 0) & dataset.contrast_eligible)
    by_episode = {}
    for index in positive:
        by_episode.setdefault(int(dataset.episode_id[index]), []).append(index)
    usable = [np.asarray(values) for values in by_episode.values() if len(values) >= 2]
    usable = usable[-config.contrast_episode_capacity :]
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


def initialize_transition_encoder(sample_dataset, config, *, pixel=False, seed=0):
    """Build the shared encoder and optimizer update for online/offline inputs."""
    key = jax.random.PRNGKey(seed)
    network = TransitionEncoder(
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
        pixel=pixel,
    )
    sample = slice(0, 1)
    params = network.init(
        key,
        jnp.asarray(sample_dataset.observation[sample]),
        jnp.asarray(sample_dataset.action[sample]),
        jnp.asarray(sample_dataset.next_observation[sample]),
        jnp.asarray(sample_dataset.terminal[sample]),
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
            prediction_valid = (
                batch[5].astype(jnp.float32)
                if len(batch) > 5
                else jnp.ones_like(target)
            )
            # Original SEA regresses the binary reward indicator directly.
            # We retain a mean reduction because JAX batches are configurable;
            # it preserves the objective without making its scale batch-sized.
            prediction_error = 0.5 * jnp.square(reward_logit - target)
            prediction_loss = (
                prediction_error * prediction_valid
            ).sum() / jnp.maximum(prediction_valid.sum(), 1.0)
            _, contrast_embeddings = network.apply(parameters, *contrast_batch)
            contrast_loss = determinant_contrast_loss(contrast_embeddings, valid)
            contrast_coefficient = config.mean_contrast_coefficient
            total = prediction_loss + contrast_coefficient * contrast_loss
            return total, (prediction_loss, contrast_loss)

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        gradient_norm = optax.global_norm(gradients)
        return state.apply_gradients(grads=gradients), (
            loss,
            metrics,
            gradient_norm,
        )

    return network, state, update


def train_transition_encoder(
    dataset: TransitionDataset,
    config: DiscoveryConfig = DiscoveryConfig(),
    *,
    pixel: bool = False,
    seed: int = 0,
    contrast_dataset: TransitionDataset | None = None,
    progress_callback=None,
):
    contrast_dataset = dataset if contrast_dataset is None else contrast_dataset
    if len(dataset) == 0:
        raise ValueError("cannot train discovery model on an empty dataset")
    rng = np.random.default_rng(seed)
    network, state, update = initialize_transition_encoder(
        dataset, config, pixel=pixel, seed=seed
    )

    metrics = None
    for update_index in range(config.train_steps):
        batch_indices = rng.integers(0, len(dataset), size=config.batch_size)
        group_indices, valid = _contrast_indices(contrast_dataset, rng, config)
        batch = (
            jnp.asarray(dataset.observation[batch_indices]),
            jnp.asarray(dataset.action[batch_indices]),
            jnp.asarray(dataset.next_observation[batch_indices]),
            jnp.asarray(dataset.terminal[batch_indices]),
            jnp.asarray(dataset.event_count[batch_indices]),
        )
        contrast_batch = (
            jnp.asarray(contrast_dataset.observation[group_indices]),
            jnp.asarray(contrast_dataset.action[group_indices]),
            jnp.asarray(contrast_dataset.next_observation[group_indices]),
            jnp.asarray(contrast_dataset.terminal[group_indices]),
        )
        state, metrics = update(state, batch, contrast_batch, jnp.asarray(valid))
        if progress_callback is not None and (
            update_index % 500 == 0 or update_index + 1 == config.train_steps
        ):
            progress_callback(update_index + 1, metrics)
    return network, state, metrics


def embed_positive_transitions(
    network, params, dataset, batch_size=1024, max_transitions=10_000
):
    indices = np.flatnonzero(dataset.event_count > 0)
    if max_transitions is not None:
        indices = indices[:max_transitions]
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


def summarize_cluster_achievements(dataset, artifacts):
    """Describe the positive prefix used by embed_positive_transitions.

    Percentages use cluster transition counts, so simultaneous unlocks can
    make their sum exceed 100. Ground-truth names are diagnostics only.
    """
    from craftax.sea.metrics import ACHIEVEMENT_NAMES

    if dataset.new_achievements is None:
        return {
            "available": False,
            "reason": "dataset has no new_achievements metadata",
        }
    indices = np.flatnonzero(dataset.event_count > 0)[: len(artifacts.labels)]
    if len(indices) != len(artifacts.labels):
        raise ValueError("cluster labels exceed positive transition count")
    achievements = np.asarray(dataset.new_achievements, dtype=bool)
    if achievements.shape != (len(dataset), len(ACHIEVEMENT_NAMES)):
        raise ValueError("new_achievements must have shape (transitions, 22)")
    selected = achievements[indices]
    clusters = []
    for cluster_id in range(len(artifacts.centroids)):
        rows = selected[artifacts.labels == cluster_id]
        count = len(rows)
        counts = rows.sum(axis=0)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "transition_count": count,
                "unlabeled_transition_count": int((~rows.any(axis=1)).sum()),
                "multi_achievement_transition_count": int((rows.sum(axis=1) > 1).sum()),
                "achievements": {
                    name: {
                        "count": int(value),
                        "percent": 100.0 * int(value) / count if count else 0.0,
                    }
                    for name, value in zip(ACHIEVEMENT_NAMES, counts)
                },
            }
        )
    return {
        "available": True,
        "transition_count": len(indices),
        "percent_denominator": "transitions in each cluster; simultaneous unlocks may sum above 100%",
        "clusters": clusters,
    }


def _count_conflicts(labels, episode_id):
    conflicts = 0
    for episode in np.unique(episode_id):
        values = labels[episode_id == episode]
        _, counts = np.unique(values, return_counts=True)
        conflicts += int(((counts * (counts - 1)) // 2).sum())
    return conflicts


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

    from sklearn.cluster import KMeans

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

    graph = infer_graph(labels, episode_id, episode_step, len(centroids)).graph
    return ClusterArtifacts(centroids, threshold, graph, labels.astype(np.int32))


__all__ = [
    "ClusterArtifacts",
    "DiscoveryConfig",
    "TransitionDataset",
    "determinant_contrast_loss",
    "embed_positive_transitions",
    "fit_clusters",
    "initialize_transition_encoder",
    "summarize_cluster_achievements",
    "train_transition_encoder",
]
