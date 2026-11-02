#!/usr/bin/env python3
"""Render a learned SEA cluster dependency graph from a run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _node_labels(report: dict, count: int) -> list[tuple[str, float]]:
    clusters = {row["cluster_id"]: row for row in report.get("clusters", [])}
    labels = []
    for cluster_id in range(count):
        row = clusters.get(cluster_id, {})
        achievements = row.get("achievements", {})
        if achievements:
            name, values = max(
                achievements.items(), key=lambda item: item[1].get("percent", 0.0)
            )
            purity = float(values.get("percent", 0.0))
            name = name.replace("_", " ").title()
        else:
            name, purity = "Unknown", 0.0
        labels.append((name, purity))
    return labels


def _layout_dag(graph: np.ndarray) -> tuple[dict[int, tuple[float, float]], list[list[int]]]:
    graph = np.asarray(graph, dtype=bool)
    count = graph.shape[0]
    if graph.shape != (count, count):
        raise ValueError("graph must be a square adjacency matrix")
    connected = graph.any(axis=0) | graph.any(axis=1)
    active = set(np.flatnonzero(connected).tolist())
    indegree = graph.sum(axis=0).astype(int)
    depth = np.zeros(count, dtype=int)
    ready = sorted(node for node in active if indegree[node] == 0)
    visited = []
    while ready:
        node = ready.pop(0)
        visited.append(node)
        for child in np.flatnonzero(graph[node]):
            depth[child] = max(depth[child], depth[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
                ready.sort()
    if len(visited) != len(active):
        cyclic = sorted(active - set(visited))
        raise ValueError(f"graph is not a DAG; cyclic cluster ids: {cyclic}")

    layers = [
        sorted(np.flatnonzero(connected & (depth == level)).tolist())
        for level in range(int(depth[connected].max()) + 1)
    ] if active else []
    isolated = np.flatnonzero(~connected).tolist()
    if isolated:
        layers.append(isolated)

    order = {node: index for layer in layers for index, node in enumerate(layer)}
    for layer_index in range(1, len(layers)):
        if isolated and layer_index == len(layers) - 1:
            break

        def barycenter(node):
            parents = np.flatnonzero(graph[:, node])
            return np.mean([order[int(parent)] for parent in parents]) if len(parents) else node

        layers[layer_index].sort(key=barycenter)
        order.update({node: index for index, node in enumerate(layers[layer_index])})

    positions = {}
    for layer_index, layer in enumerate(layers):
        xs = np.arange(len(layer), dtype=float) - (len(layer) - 1) / 2
        for x, node in zip(xs, layer):
            positions[node] = (float(x), float(-layer_index))
    return positions, layers


def render_graph(run_dir: Path, output: Path, title: str | None = None, dpi: int = 180):
    clusters_path = run_dir / "clusters.npz"
    report_path = run_dir / "cluster_achievements.json"
    if not clusters_path.exists():
        raise FileNotFoundError(f"missing {clusters_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"missing {report_path}")

    with np.load(clusters_path) as data:
        graph = np.asarray(data["graph"], dtype=bool)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("available", False):
        raise ValueError(f"cluster achievement labels unavailable: {report.get('reason')}")

    labels = _node_labels(report, len(graph))
    positions, layers = _layout_dag(graph)
    max_width = max((len(layer) for layer in layers), default=1)
    figure_width = max(10.0, 3.15 * max_width)
    figure_height = max(4.0, 1.3 * len(layers))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.set_facecolor("#fbfcff")
    fig.patch.set_facecolor("#fbfcff")

    node_width, node_height = 2.55, 0.58
    x_spacing, y_spacing = 3.05, 1.12
    scaled = {node: (x * x_spacing, y * y_spacing) for node, (x, y) in positions.items()}

    for source, target in np.argwhere(graph):
        x1, y1 = scaled[int(source)]
        x2, y2 = scaled[int(target)]
        arrow = FancyArrowPatch(
            (x1, y1 - node_height / 2),
            (x2, y2 + node_height / 2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#7192cf",
            connectionstyle="arc3,rad=0.0",
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
        ax.add_patch(arrow)

    for node, (x, y) in scaled.items():
        name, purity = labels[node]
        strength = np.clip(purity / 100.0, 0.0, 1.0)
        face = (0.91 - 0.08 * strength, 0.94 - 0.05 * strength, 0.99, 1.0)
        box = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.08,rounding_size=0.22",
            linewidth=1.6,
            edgecolor="#6688c8",
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            f"C{node} · {name}\nPurity {purity:.1f}%",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#263b66",
            fontweight="semibold",
            zorder=3,
        )

    if scaled:
        xs, ys = zip(*scaled.values())
        ax.set_xlim(min(xs) - node_width, max(xs) + node_width)
        ax.set_ylim(min(ys) - node_height, max(ys) + node_height)
    ax.axis("off")
    ax.set_title(
        title or f"SEA Achievement Dependency Graph (K={len(graph)})",
        fontsize=16,
        color="#263b66",
        pad=20,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="SEA result directory containing clusters.npz")
    parser.add_argument(
        "--output",
        help="output .png/.pdf path; defaults to RUN_DIR/sea_graph.png",
    )
    parser.add_argument("--title", help="optional figure title")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "sea_graph.png"
    )
    render_graph(run_dir, output, args.title, args.dpi)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
