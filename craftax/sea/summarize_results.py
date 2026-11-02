"""Summarize final-window PPO training rollouts for SEA comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from craftax.sea.metrics import (
    ACHIEVEMENT_NAMES,
    EASY17_ACHIEVEMENT_INDICES,
    HARD_ACHIEVEMENT_INDICES,
    NO_EAT_PLANT_ACHIEVEMENT_INDICES,
)


def summarize_history(path: str | Path, window: int = 50) -> dict:
    """Aggregate completed episodes from the final ``window`` PPO updates."""

    path = Path(path)
    if window < 1:
        raise ValueError("window must be positive")
    with np.load(path) as history:
        if "episode_count" not in history or "achievement_success_count" not in history:
            raise ValueError(f"{path} does not contain episode metrics")
        episode_count = np.asarray(history["episode_count"])
        achievement_count = np.asarray(history["achievement_success_count"])

    if episode_count.ndim != 1:
        raise ValueError("episode_count must be a vector of PPO updates")
    if achievement_count.shape != (len(episode_count), len(ACHIEVEMENT_NAMES)):
        raise ValueError(
            "achievement_success_count must have shape "
            f"(updates, {len(ACHIEVEMENT_NAMES)})"
        )

    used_updates = min(window, len(episode_count))
    episodes = int(episode_count[-used_updates:].sum())
    if episodes == 0:
        raise ValueError(f"{path} has no completed episodes in its final window")
    counts = achievement_count[-used_updates:].sum(axis=0)
    rates = counts / episodes
    score21_rates = rates[list(NO_EAT_PLANT_ACHIEVEMENT_INDICES)]
    clusters_path = path.parent / "clusters.npz"
    num_clusters = None
    if path.name.startswith("goal_") and clusters_path.exists():
        with np.load(clusters_path) as clusters:
            if "centroids" in clusters:
                num_clusters = int(clusters["centroids"].shape[0])

    return {
        "history": str(path.resolve()),
        "stage": "goal" if path.name.startswith("goal_") else "base",
        "num_clusters": num_clusters,
        "available_updates": int(len(episode_count)),
        "window_updates": used_updates,
        "completed_episodes": episodes,
        "achievements_per_episode21": float(score21_rates.sum()),
        "easy17_success_rate": float(
            rates[list(EASY17_ACHIEVEMENT_INDICES)].mean()
        ),
        "hard4_success_rate": float(rates[list(HARD_ACHIEVEMENT_INDICES)].mean()),
        "crafter_score21": float(
            np.exp(np.log1p(100.0 * score21_rates).mean()) - 1.0
        ),
        "achievement_success_rate": {
            name: float(rate) for name, rate in zip(ACHIEVEMENT_NAMES, rates)
        },
        "achievement_success_count": {
            name: int(count) for name, count in zip(ACHIEVEMENT_NAMES, counts)
        },
    }


def find_histories(inputs: list[str], stage: str = "auto") -> list[Path]:
    """Resolve history files, preferring goal over base for each run in auto mode."""

    names = {
        "base": ("base_metrics_history.npz",),
        "goal": ("goal_metrics_history.npz",),
        "auto": ("goal_metrics_history.npz", "base_metrics_history.npz"),
    }[stage]
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            resolved.append(path.resolve())
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)

        run_dirs: dict[Path, dict[str, Path]] = {}
        for name in names:
            for candidate in path.rglob(name):
                run_dirs.setdefault(candidate.parent.resolve(), {})[name] = candidate
        for files in run_dirs.values():
            selected = next((files[name] for name in names if name in files), None)
            if selected is not None:
                resolved.append(selected.resolve())

    unique = sorted(set(resolved))
    if not unique:
        raise FileNotFoundError(
            f"no {stage} metrics history found under: {', '.join(inputs)}"
        )
    return unique


def aggregate_runs(runs: list[dict]) -> dict:
    """Return an unweighted across-run mean and sample standard deviation."""

    keys = (
        "achievements_per_episode21",
        "easy17_success_rate",
        "hard4_success_rate",
        "crafter_score21",
    )
    result = {"num_runs": len(runs)}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def _print_report(runs: list[dict], aggregate: dict) -> None:
    columns = (
        ("run", 16),
        ("stage", 7),
        ("updates", 9),
        ("K", 4),
        ("Easy17(%)", 11),
        ("Hard4(%)", 10),
        ("Score21", 9),
        ("Diamond(%)", 12),
        ("IronPickaxe(%)", 16),
    )
    print(" ".join(name.ljust(width) for name, width in columns))
    print(" ".join("-" * width for _, width in columns))
    for run in runs:
        rates = run["achievement_success_rate"]
        label = Path(run["history"]).parent.name
        values = (
            label.ljust(16),
            run["stage"].ljust(7),
            str(run["window_updates"]).rjust(9),
            ("-" if run["num_clusters"] is None else str(run["num_clusters"])).rjust(4),
            f"{100 * run['easy17_success_rate']:.2f}".rjust(11),
            f"{100 * run['hard4_success_rate']:.2f}".rjust(10),
            f"{run['crafter_score21']:.2f}".rjust(9),
            f"{100 * rates['collect_diamond']:.2f}".rjust(12),
            f"{100 * rates['make_iron_pickaxe']:.2f}".rjust(16),
        )
        print(" ".join(values))

    if len(runs) > 1:
        def mean_std(key: str, scale: float = 1.0) -> str:
            values = aggregate[key]
            return f"{scale * values['mean']:.2f} +/- {scale * values['std']:.2f}"

        print("\nAcross-run mean +/- sample std (runs are weighted equally)")
        print(f"Easy17(%): {mean_std('easy17_success_rate', 100)}")
        print(f"Hard4(%): {mean_std('hard4_success_rate', 100)}")
        print(f"Score21: {mean_std('crafter_score21')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the final PPO-update window from base/goal training "
            "rollouts. This is not held-out evaluation."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="run directories, a run root containing seed-* dirs, or history .npz files",
    )
    parser.add_argument("--window", type=int, default=50, help="final PPO updates")
    parser.add_argument(
        "--stage",
        choices=("auto", "base", "goal"),
        default="auto",
        help="auto prefers goal history when both stages exist in one run",
    )
    parser.add_argument("--output", help="optional JSON report path")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    histories = find_histories(args.inputs, args.stage)
    runs = [summarize_history(path, args.window) for path in histories]
    aggregate = aggregate_runs(runs)
    _print_report(runs, aggregate)

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "protocol": {
                "kind": "final-window training rollout",
                "held_out_evaluation": False,
                "window_updates": args.window,
                "achievement_count": 21,
                "excluded_achievements": ["eat_plant"],
                "easy_set": "Easy17",
                "hard_set": "Hard4",
            },
            "runs": runs,
            "aggregate": aggregate,
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
