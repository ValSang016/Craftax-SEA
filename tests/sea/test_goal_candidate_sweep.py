"""Regression checks for failed-run preservation and seed-level statistics."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("analyze_goal_candidate_sweep")
runner = load_script("run_goal_candidate_sweep")


def test_one_seed_does_not_create_a_t_test_or_confidence_interval():
    result = analysis.paired_statistics([0.693])
    assert result["n_pairs"] == 1
    assert result["paired_t_p"] is None
    assert result["ci95_low"] is None
    assert result["exact_sign_flip_p"] == 1.0


def test_three_pairs_use_seed_variability_and_all_eight_label_swaps():
    result = analysis.paired_statistics([1, 2, 3])
    # For df=2, two-sided t p = 1 - t / sqrt(t**2 + 2).
    assert result["paired_t_p"] == pytest.approx(1 - (6 / 7) ** 0.5)
    assert result["exact_sign_flip_p"] == 0.25
    assert result["mean_difference"] == 2
    assert result["ci95_low"] < 0 < result["ci95_high"]


def test_holm_keeps_three_comparisons_when_one_is_missing():
    assert analysis.holm_adjust([None, 0.01, 0.2]) == [None, 0.03, 0.4]
    assert analysis.holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_score_pools_episode_counts_instead_of_averaging_update_scores(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    episodes = np.zeros(4577)
    episodes[-2:] = [1, 9]
    achievements = np.zeros((4577, 22))
    achievements[-2, 0] = 1
    np.savez(directory / "goal_metrics_history.npz", episode_count=episodes,
             achievement_success_count=achievements)
    manifest = dict(jobs=[dict(state="completed", output=str(directory),
                              condition="all-once", beta=0.8, seed=0)])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, rows = analysis.collect_scores(tmp_path)
    assert rows[0]["episodes"] == 10
    assert rows[0]["achievements_per_episode"] == 0.1
    assert rows[0]["crafter_score"] == pytest.approx(np.expm1(np.log(11) / 22))


def test_failed_retry_archives_failures_and_preserves_completed_results(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    frozen = source / "module.py"
    frozen.write_text("original training source")
    graph = tmp_path / "oracle_graph.npz"
    graph.write_bytes(b"original graph")
    completed = tmp_path / "complete"
    completed.mkdir()
    for filename in ("goal_metrics.json", "goal_metrics_history.npz"):
        (completed / filename).write_bytes(b"finished result")
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "train.log").write_text("original CUDA failure")
    manifest = dict(source_snapshot=str(source),
        source_sha256={"module.py": hashlib.sha256(frozen.read_bytes()).hexdigest()},
        graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(), jobs=[
            dict(id=0, state="completed", output=str(completed)),
            dict(id=1, state="failed", output=str(failed), exit_code=1),
        ])
    runner.write_json(tmp_path / "manifest.json", manifest)
    loaded = runner.load_resume_plan(tmp_path)
    runner.requeue_failed_jobs(tmp_path, loaded)
    assert loaded["jobs"][0] == manifest["jobs"][0]
    assert (completed / "goal_metrics_history.npz").read_bytes() == b"finished result"
    assert loaded["jobs"][1]["state"] == "queued"
    archived = Path(loaded["jobs"][1]["previous_attempts"][0]["archived_output"])
    assert (archived / "train.log").read_text() == "original CUDA failure"
    assert not failed.exists()
    frozen.write_text("unexpected source change")
    # Restore a settled manifest so source validation is reached.
    runner.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source changed"):
        runner.load_resume_plan(tmp_path)
