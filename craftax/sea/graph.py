"""SEA temporal-precedence graph inference shared by learned and oracle labels.

Edges encode observed ordering, not causal necessity. Thresholds and reduction
preserve the fix123 algorithm, including within-episode duplicate counts.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GraphInference:
    graph: np.ndarray
    candidates: np.ndarray
    order: np.ndarray
    happens: np.ndarray


def transitive_reduction(graph):
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


def infer_graph(labels, episode_id, episode_step, num_nodes):
    """Infer the fixed >.97 forward / <.001 reverse temporal graph.

    Count every ordered event pair, including repeated labels. Rows within an
    episode retain the original argsort(step) behavior. Group once to avoid a
    full dataset scan per episode; this also supports million-event datasets.
    """
    labels, episode_id, episode_step = map(
        np.asarray, (labels, episode_id, episode_step)
    )
    if (
        labels.ndim != 1
        or labels.shape != episode_id.shape
        or labels.shape != episode_step.shape
    ):
        raise ValueError("labels, episode_id and episode_step must be aligned vectors")
    if num_nodes < 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("positive node count and integer labels required")
    if np.any(labels < 0) or np.any(labels >= num_nodes):
        raise ValueError("label outside node range")
    order = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    happens = np.bincount(labels, minlength=num_nodes).astype(np.int64)
    if len(labels):
        indexes = np.argsort(episode_id, kind="stable")
        boundaries = np.flatnonzero(np.diff(episode_id[indexes])) + 1
        for group in np.split(indexes, boundaries):
            group = group[np.argsort(episode_step[group])]
            sequence = labels[group]
            for position, source in enumerate(sequence):
                np.add.at(order[source], sequence[position + 1 :], 1)
    forward = order / np.maximum(1, happens)[None, :]
    reverse = order.T / np.maximum(1, order)
    candidates = ((order > 0) & (forward > 0.97) & (reverse < 0.001)).astype(np.int8)
    return GraphInference(transitive_reduction(candidates), candidates, order, happens)
