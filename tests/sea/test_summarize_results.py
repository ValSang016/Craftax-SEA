import numpy as np

from craftax.sea.metrics import ACHIEVEMENT_NAMES
from craftax.sea.summarize_results import (
    aggregate_runs,
    find_histories,
    summarize_history,
)


def _write_history(path, episode_count, rates):
    counts = np.asarray(episode_count)[:, None] * np.asarray(rates)[None, :]
    np.savez_compressed(
        path,
        episode_count=np.asarray(episode_count, dtype=np.int32),
        achievement_success_count=counts.astype(np.int32),
    )


def test_summarize_history_uses_only_final_window_and_excludes_eat_plant(tmp_path):
    rates = np.zeros(len(ACHIEVEMENT_NAMES))
    rates[:] = 0.5
    rates[ACHIEVEMENT_NAMES.index("eat_plant")] = 1.0
    path = tmp_path / "base_metrics_history.npz"
    _write_history(path, [10, 20, 20], rates)

    result = summarize_history(path, window=2)

    assert result["completed_episodes"] == 40
    assert result["window_updates"] == 2
    assert result["easy17_success_rate"] == 0.5
    assert result["hard4_success_rate"] == 0.5
    assert result["achievements_per_episode21"] == 10.5
    assert np.isclose(result["crafter_score21"], 50.0)
    assert result["num_clusters"] is None


def test_goal_summary_reads_cluster_count(tmp_path):
    rates = np.full(len(ACHIEVEMENT_NAMES), 0.5)
    path = tmp_path / "goal_metrics_history.npz"
    _write_history(path, [20], rates)
    np.savez_compressed(tmp_path / "clusters.npz", centroids=np.zeros((17, 256)))

    result = summarize_history(path)

    assert result["stage"] == "goal"
    assert result["num_clusters"] == 17


def test_find_histories_auto_prefers_goal_per_run(tmp_path):
    seed0 = tmp_path / "seed-0"
    seed1 = tmp_path / "seed-1"
    seed0.mkdir()
    seed1.mkdir()
    for path in (
        seed0 / "base_metrics_history.npz",
        seed0 / "goal_metrics_history.npz",
        seed1 / "base_metrics_history.npz",
    ):
        path.touch()

    found = find_histories([str(tmp_path)], stage="auto")

    assert found == sorted(
        [
            (seed0 / "goal_metrics_history.npz").resolve(),
            (seed1 / "base_metrics_history.npz").resolve(),
        ]
    )


def test_aggregate_runs_uses_equal_seed_weighting_and_sample_std():
    runs = [
        {
            "achievements_per_episode21": 1.0,
            "easy17_success_rate": 0.2,
            "hard4_success_rate": 0.3,
            "crafter_score21": 4.0,
        },
        {
            "achievements_per_episode21": 3.0,
            "easy17_success_rate": 0.4,
            "hard4_success_rate": 0.7,
            "crafter_score21": 8.0,
        },
    ]

    result = aggregate_runs(runs)

    assert result["num_runs"] == 2
    assert result["crafter_score21"]["mean"] == 6.0
    assert np.isclose(result["crafter_score21"]["std"], np.sqrt(8.0))
