"""Check selector ablations without launching training runs or rollouts."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from craftax.sea.dynamic_graph import (
    DynamicGraphConfig,
    compute_dynamic_edge_scores,
    get_goal_candidates,
    select_high_level_goal,
)
from craftax.sea.train import _make_dynamic_config, build_parser
from craftax.sea.train_goal_only import (
    _resolve_dynamic_config,
    build_parser as goal_only_parser,
)


def test_zero_lambda_removes_graph_scores_but_keeps_completed_mask():
    graph = jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    completed = jnp.asarray([[True, False, False]])
    full, base = jnp.asarray([[4., 3., 5.]]), jnp.asarray([[2., 1., 4.]])
    score = jax.jit(lambda g: compute_dynamic_edge_scores(
        g, jnp.asarray([0]), full, base, .8, structural_prior_weight=0.,
    ))
    np.testing.assert_array_equal(score(graph), score(jnp.zeros_like(graph)))
    candidates = get_goal_candidates(
        completed, graph, False, allow_unready_goals=True, allow_completed_goals=False,
    )
    _, probabilities = select_high_level_goal(
        jax.random.PRNGKey(2), candidates, score(graph), 1., False,
    )
    assert probabilities[0, 0] == 0
    assert probabilities[0, 2] > 0  # Unready is still eligible after warmup.
    np.testing.assert_allclose(probabilities[0, 1] / probabilities[0, 2], np.exp(.8), rtol=1e-6)
    warmup_mask = get_goal_candidates(
        completed, graph, False, allow_unready_goals=True,
        allow_completed_goals=False, warmup=True,
    )
    np.testing.assert_array_equal(warmup_mask, [[False, True, False]])


def test_base_q_signal_ignores_full_q_and_preserves_softmax_scale():
    graph = jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    base = jnp.asarray([[4., 1., 5.]])
    score = jax.jit(lambda full: compute_dynamic_edge_scores(
        graph, jnp.asarray([0]), full, base, .3,
        structural_prior_weight=1., score_source="base-q",
    ))
    logits = score(jnp.zeros_like(base))
    np.testing.assert_array_equal(logits, score(jnp.full_like(base, 10000.)))
    _, probabilities = select_high_level_goal(
        jax.random.PRNGKey(3), jnp.asarray([[False, True, True]]), logits, 1., False,
    )
    # A_1=1, Q_B1=1; A_2=0, Q_B2=5: logit difference = 1.5 - 1.3.
    np.testing.assert_allclose(probabilities[0, 2], 1 / (1 + np.exp(-.2)), rtol=1e-6)
    assert probabilities[0, 2] > probabilities[0, 1]


def test_full_q_signal_ignores_base_q_and_uses_absolute_return():
    graph = jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    full = jnp.asarray([[4., 1., 5.]])
    score = jax.jit(lambda base: compute_dynamic_edge_scores(
        graph, jnp.asarray([0]), full, base, .3,
        structural_prior_weight=1., score_source="full-q",
    ))
    logits = score(jnp.zeros_like(full))
    np.testing.assert_allclose(logits, [[1.2, 1.3, 1.5]], rtol=1e-6)
    np.testing.assert_array_equal(logits, score(jnp.asarray([[100., 20., -100.]])))
    _, probabilities = select_high_level_goal(
        jax.random.PRNGKey(3), jnp.asarray([[False, True, True]]), logits, 1., False,
    )
    np.testing.assert_allclose(probabilities[0, 2], 1 / (1 + np.exp(-.2)), rtol=1e-6)


def test_zero_beta_uses_only_scaled_prior_and_unknown_node_has_no_prior():
    graph = jnp.asarray([[0, 1], [0, 0]])
    full, base = jnp.asarray([[3., 4., 8.]]), jnp.asarray([[1., 2., 3.]])
    for signal in ("residual", "base-q", "full-q"):
        scores = compute_dynamic_edge_scores(
            graph, jnp.asarray([0]), full, base, 0.,
            structural_prior_weight=2., score_source=signal,
        )
        np.testing.assert_array_equal(scores, [[0., 2., 0.]])
        scores = compute_dynamic_edge_scores(
            graph, jnp.asarray([2]), full, base, 0.,
            structural_prior_weight=2., score_source=signal,
        )
        np.testing.assert_array_equal(scores, [[0., 0., 0.]])


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_invalid_structural_prior_weight_is_rejected_before_training(weight):
    args = build_parser().parse_args(["--symbolic", "--dynamic-lambda", str(weight)])
    with pytest.raises(ValueError, match="structural prior"):
        _make_dynamic_config(args)


def test_scoring_settings_reach_serialized_runtime_config_and_legacy_defaults():
    parser = build_parser()
    original = _make_dynamic_config(parser.parse_args(["--symbolic"]))
    assert original.structural_prior_weight == 1.
    assert original.score_source == "residual"
    args = parser.parse_args([
        "--symbolic", "--dynamic-lambda", "0", "--dynamic-score-source", "base-q",
        "--dynamic-beta", "0.2", "--allow-unready-goals", "true", "--allow-completed-goals", "false",
    ])
    config = _make_dynamic_config(args)
    saved = json.loads(json.dumps(asdict(config)))
    assert saved["structural_prior_weight"] == 0.
    assert saved["score_source"] == "base-q"
    assert saved["residual_weight"] == .2
    assert saved["allow_unready_goals"] and not saved["allow_completed_goals"]
    full_args = parser.parse_args(["--symbolic", "--dynamic-score-source", "full-q", "--beta", ".5"])
    full_config = _make_dynamic_config(full_args)
    assert full_config.score_source == "full-q" and full_config.residual_weight == .5
    with pytest.raises(ValueError, match="require --symbolic"):
        _make_dynamic_config(parser.parse_args(["--dynamic-lambda", "0"]))
    with pytest.raises(ValueError, match="score source"):
        DynamicGraphConfig(score_source="typo").validate()


def test_goal_only_inherits_old_defaults_and_new_settings_with_explicit_overrides():
    parser = goal_only_parser()
    inherited = parser.parse_args(["source", "--output", "unused"])
    old = _resolve_dynamic_config({"dynamic_graph": {"residual_weight": .8}}, inherited, False)
    assert (old.structural_prior_weight, old.score_source, old.residual_weight) == (1., "residual", .8)
    source = {"dynamic_graph": asdict(DynamicGraphConfig(
        residual_weight=.5, structural_prior_weight=0., score_source="base-q",
        allow_unready_goals=True,
    ))}
    saved = _resolve_dynamic_config(source, inherited, False)
    assert (saved.structural_prior_weight, saved.score_source, saved.residual_weight) == (0., "base-q", .5)
    overridden = parser.parse_args([
        "source", "--output", "unused", "--dynamic-lambda", "1",
        "--dynamic-score-source", "residual", "--dynamic-beta", ".2",
    ])
    config = _resolve_dynamic_config(source, overridden, False)
    assert (config.structural_prior_weight, config.score_source, config.residual_weight) == (1., "residual", .2)
    assert config.allow_unready_goals and not config.allow_completed_goals
    overridden.dynamic_score_source = "full-q"
    assert _resolve_dynamic_config(source, overridden, False).score_source == "full-q"


@pytest.mark.parametrize("suite,count", [("selector-ablations", 15), ("full-q", 9)])
def test_selector_suite_only_prepares_exact_commands_without_execute(tmp_path, monkeypatch, suite, count):
    path = Path(__file__).resolve().parents[2] / "scripts/run_goal_candidate_sweep.py"
    spec = importlib.util.spec_from_file_location("selector_sweep", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    fake_repo = tmp_path / "repo"
    (fake_repo / "craftax").mkdir(parents=True)
    (fake_repo / "craftax/__init__.py").write_text("# frozen source")
    reference = tmp_path / "reference.json"
    saved = vars(build_parser().parse_args(["--symbolic"]))
    saved.update(wandb_project="test", wandb_entity="test")
    reference.write_text(json.dumps({"args": saved}))
    graph = tmp_path / "graph.npz"
    graph.write_bytes(b"frozen graph")
    output = tmp_path / "planned"
    monkeypatch.setattr(runner, "REPO", fake_repo)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda argv, **kwargs: "revision" if "rev-parse" in argv else "")
    def forbidden(*args, **kwargs):
        pytest.fail("training/scheduling must not start without --execute")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    monkeypatch.setattr(runner, "summarize", forbidden)
    monkeypatch.setattr("sys.argv", [str(path), "--output", str(output), "--suite", suite, "--gpus", "0", "1",
                                     "--reference-config", str(reference), "--graph", str(graph)])
    runner.main()
    manifest = runner.load_prepared_plan(output)
    jobs = manifest["jobs"]
    assert len(jobs) == count and manifest["suite"] == suite
    assert manifest["gpu_ids"] == [0, 1]
    settings = runner.SELECTOR_ABLATIONS if suite == "selector-ablations" else [
        ("full-q-with-prior", beta, 1., "full-q") for beta in (.2, .3, .5)]
    expected = {(seed, weight, signal, beta) for seed in (0, 1, 2)
                for _, beta, weight, signal in settings}
    assert {(j["seed"], j["structural_prior_weight"], j["score_source"], j["beta"]) for j in jobs} == expected
    assert len({j["output"] for j in jobs}) == count
    for job in jobs:
        assert job["state"] == "queued" and not Path(job["output"]).exists()
        args = build_parser().parse_args(job["command"][4:])
        config = _make_dynamic_config(args)
        assert config.allow_unready_goals and not config.allow_completed_goals
        assert config.structural_prior_weight == job["structural_prior_weight"]
        assert config.score_source == job["score_source"]
        assert config.residual_weight == job["beta"]
    # Starting a prepared plan without explicit GPUs inherits its saved 0/1 list.
    monkeypatch.setattr("sys.argv", [str(path), "--output", str(output), "--start-prepared"])
    runner.main()
    frozen = Path(manifest["source_snapshot"]) / "craftax/__init__.py"
    frozen.write_text("# changed after review")
    with pytest.raises(ValueError, match="source changed"):
        runner.load_prepared_plan(output)
