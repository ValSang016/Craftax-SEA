"""Behavioral checks for temporal graph inference, independent of embeddings."""

import numpy as np
from craftax.sea.graph import infer_graph


def test_reduction_preserves_chain_and_unobserved_nodes():
    result = infer_graph(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([1, 2, 3, 1, 2, 3]),
        4,
    )
    assert result.candidates[0, 2] == 1
    np.testing.assert_array_equal(
        result.graph, [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(result.happens, [2, 2, 2, 0])


def test_forward_threshold_is_strict_and_uses_all_target_events():
    # 97 co-occurrences among 100 B events is exactly .97: must not pass.
    labels = np.array([0, 1] * 97 + [1] * 3)
    episodes = np.array([i for i in range(97) for _ in range(2)] + [97, 98, 99])
    steps = np.array([0, 1] * 97 + [1] * 3)
    result = infer_graph(labels, episodes, steps, 2)
    assert result.order[0, 1] == 97 and result.happens[1] == 100
    assert result.graph.sum() == 0


def test_repeated_labels_are_counted_as_events_not_episode_presence():
    r = infer_graph(np.array([0, 1, 1]), np.array([4, 4, 4]), np.array([0, 1, 2]), 2)
    np.testing.assert_array_equal(r.order, [[0, 2], [0, 1]])
    np.testing.assert_array_equal(r.happens, [1, 2])
    assert r.graph[0, 1] == 1
