import jax.numpy as jnp
import numpy as np

from craftax.sea.discovery import determinant_contrast_loss, fit_clusters
from craftax.sea.goal import classify_embeddings


def test_determinant_loss_is_finite_for_identical_embeddings():
    embeddings = jnp.zeros((2, 3, 4))
    valid = jnp.array([[True, True, True], [True, True, False]])
    assert jnp.isfinite(determinant_contrast_loss(embeddings, valid))


def test_cluster_artifacts_and_unknown_classification():
    embeddings = np.array(
        [[0.0, 0.0], [0.1, 0.0], [4.9, 5.0], [5.0, 5.1]], dtype=np.float32
    )
    episode_id = np.array([0, 1, 0, 1])
    episode_step = np.array([1, 1, 2, 2])
    artifacts = fit_clusters(
        embeddings,
        episode_id,
        episode_step,
        min_clusters=2,
        max_clusters=2,
    )
    assert artifacts.centroids.shape == (2, 2)
    assert artifacts.graph.shape == (2, 2)
    classes, _ = classify_embeddings(
        jnp.asarray([[100.0, 100.0]]),
        jnp.asarray(artifacts.centroids),
        artifacts.threshold,
    )
    assert classes[0] == 2
