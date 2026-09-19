import jax.numpy as jnp
import numpy as np

from craftax.sea.discovery import (
    DiscoveryConfig,
    determinant_contrast_loss,
    fit_clusters,
)
from craftax.sea.goal import classify_embeddings


def test_default_representation_dimensions_match_original_sea():
    config = DiscoveryConfig()
    assert config.hidden_size == 256
    assert config.embedding_size == 256
    assert config.contrast_groups == 128
    assert config.contrast_episode_capacity == 256


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


def test_cluster_achievement_report_alignment_and_legacy(tmp_path):
    from craftax.sea.discovery import (
        ClusterArtifacts, TransitionDataset, summarize_cluster_achievements,
    )
    achievements = np.zeros((5, 22), dtype=bool)
    achievements[1, 0] = True
    achievements[2, [0, 19]] = True
    achievements[4, 19] = True
    dataset = TransitionDataset(
        observation=np.zeros((5, 2)), action=np.zeros(5),
        next_observation=np.zeros((5, 2)), terminal=np.zeros(5, dtype=bool),
        event_count=np.array([0, 1, 2, 0, 1]), episode_id=np.arange(5),
        episode_step=np.zeros(5), contrast_eligible=np.ones(5, dtype=bool),
        new_achievements=achievements,
    )
    # Only the first two positive transitions were embedded; exclude row 4.
    artifacts = ClusterArtifacts(np.zeros((2, 2)), 1.0, np.zeros((2, 2)), np.array([0, 0]))
    path = tmp_path / 'dataset.npz'
    dataset.save(path)
    report = summarize_cluster_achievements(TransitionDataset.load(path), artifacts)
    assert report['transition_count'] == 2
    cluster = report['clusters'][0]
    assert cluster['achievements']['collect_wood'] == {'count': 2, 'percent': 100.0}
    assert cluster['achievements']['collect_diamond'] == {'count': 1, 'percent': 50.0}
    assert cluster['multi_achievement_transition_count'] == 1
    assert report['clusters'][1]['transition_count'] == 0
    dataset.new_achievements = None
    dataset.save(path)
    legacy = TransitionDataset.load(path)
    assert legacy.new_achievements is None
    assert not summarize_cluster_achievements(legacy, artifacts)['available']
