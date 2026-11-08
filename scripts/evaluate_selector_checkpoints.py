"""Evaluate frozen selector checkpoints on fixed, complete held-out episodes."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def prepare(args):
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"evaluation output already exists: {root}")
    latest = (args.training_sweep.resolve() if args.training_sweep else
              REPO / "runs/oracle-selector-ablations-prepared-20261108-145834")
    previous = REPO / "runs/oracle-goal-eligibility-first-reward-20261105-233352"
    source = json.loads((latest / "manifest.json").read_text())
    controls = json.loads((previous / "manifest.json").read_text()) if not args.training_sweep else {"jobs": []}
    jobs = []
    for job in source["jobs"] + [j for j in controls["jobs"] if j["condition"] == "all-once" and j["beta"] in (.8, 10.)]:
        if job["state"] != "completed":
            raise ValueError("all selected training checkpoints must be complete")
        name = job["condition"] if job["condition"] != "all-once" else "residual-with-prior"
        name += "-beta-" + str(job["beta"]).replace(".", "p")
        record = dict(id=len(jobs), condition=name, training_seed=job["seed"],
            source=job["output"], protocol="training-env")
        jobs.append(record)
    if not args.training_sweep:
        for j in list(jobs):
            if j["condition"] in ("base-q-with-prior-beta-0p5", "residual-with-prior-beta-10p0"):
                jobs.append(dict(j, id=len(jobs), protocol="sea-style-termination"))
    root.mkdir(parents=True)
    evaluation_source = Path(source["source_snapshot"]) if args.training_sweep else REPO
    if args.training_sweep:
        for relative, expected in source["source_sha256"].items():
            if hashlib.sha256((evaluation_source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"frozen training source changed: {relative}")
    shutil.copytree(evaluation_source / "craftax", root / "source/craftax", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, root / "evaluator-runner.py")
    for job in jobs:
        directory = Path(job["source"])
        job["checkpoint_sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("goal_policy.msgpack", "high_level_q.msgpack", "config.json", "oracle_graph.npz")}
        job["output"] = str(root / job["protocol"] / job["condition"] / f"seed-{job['training_seed']}")
        job["gpu"] = args.gpus[job["id"] % len(args.gpus)]
    assert len(jobs) == (len(source["jobs"]) if args.training_sweep else 27)
    manifest = dict(created_at_kst=now(), episodes_per_checkpoint=args.episodes,
        environment_seed_start=args.seed_start, policy_seed=args.policy_seed, block=args.block,
        gpus=args.gpus, training_updates=False, jobs=jobs,
        protocol="fixed independent cohort; all episodes complete; all 22 achievement columns retained",
        source_sha256={str(p.relative_to(root / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((root / "source").rglob("*.py"))})
    write_json(root / "manifest.json", manifest)
    return manifest


def worker(root, gpu):
    import csv
    import jax
    import jax.numpy as jnp
    import numpy as np
    from craftax.sea.evaluation import build_evaluator, restore, summarize_episodes
    from craftax.sea.goal import load_oracle_graph
    from craftax.sea.metrics import ACHIEVEMENT_NAMES

    manifest = json.loads((root / "manifest.json").read_text())
    cache = {}
    status_path = root / f"worker-{gpu}.json"
    for job in [j for j in manifest["jobs"] if j["gpu"] == gpu]:
        directory = Path(job["output"])
        if (directory / "summary.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(job["source"])
        for name, expected in job["checkpoint_sha256"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"source checkpoint changed: {source / name}")
        config = json.loads((source / "config.json").read_text())
        signature = (job["protocol"], config["goal_ppo"]["hidden_size"],
            config["args"]["idle_timeout"], config["args"]["immortal"],
            config["args"]["health_reward_coef"], config["args"]["combat_damage_mode"],
            job["checkpoint_sha256"]["oracle_graph.npz"],
            config["dynamic_graph"]["temperature"], config["dynamic_graph"]["goal_timeout"],
            config["dynamic_graph"].get("allow_unready_goals", False),
            config["dynamic_graph"].get("allow_completed_goals", False))
        if signature not in cache:
            cache[signature] = build_evaluator(config, load_oracle_graph(source / "oracle_graph.npz"),
                episodes=manifest["episodes_per_checkpoint"], block=manifest["block"], protocol=job["protocol"])
        initialize, rollout, protocol_info = cache[signature]
        policy = restore(source / "goal_policy.msgpack")
        q = restore(source / "high_level_q.msgpack")
        state, initial_hash = initialize(manifest["environment_seed_start"], manifest["policy_seed"])
        dynamic = config["dynamic_graph"]
        started = time.monotonic()
        for start in range(0, protocol_info["max_timesteps"], manifest["block"]):
            state = rollout(state, policy, q["full_q"], q["base_q"],
                jnp.float32(dynamic["residual_weight"]), jnp.float32(dynamic.get("structural_prior_weight", 1.)),
                jnp.int32(("residual", "base-q", "full-q").index(dynamic.get("score_source", "residual"))))
            active = int(np.asarray(jax.device_get(state.active)).sum())
            progress = dict(updated_at_kst=now(), job=job, step=int(np.asarray(state.step)),
                completed_episodes=manifest["episodes_per_checkpoint"]-active,
                target_episodes=manifest["episodes_per_checkpoint"], active=active,
                elapsed_seconds=time.monotonic()-started)
            write_json(status_path, progress)
            if start == 0 or start % 1024 == 0 or active == 0:
                print(now(), "progress", job["id"], job["condition"], job["protocol"], progress["step"], active, flush=True)
            if active == 0:
                break
        arrays = {name: np.asarray(jax.device_get(getattr(state, name))) for name in (
            "achievements", "lengths", "terminal_count", "terminal_reason", "first_achievement_steps",
            "selected_goal_counts", "goal_success_count", "environment_return", "active")}
        assert (arrays["terminal_count"] == 1).all(), "every planned episode must finish exactly once"
        assert (arrays["lengths"] >= 1).all() and (arrays["lengths"] <= protocol_info["max_timesteps"]).all()
        assert np.array_equal(arrays["first_achievement_steps"] >= 0, arrays["achievements"])
        seeds = np.arange(manifest["environment_seed_start"], manifest["environment_seed_start"]+manifest["episodes_per_checkpoint"])
        np.savez_compressed(directory / "episodes.npz", evaluation_seeds=seeds, **arrays)
        with (directory / "episodes.csv").open("w", newline="") as handle:
            names = ["evaluation_seed", "episode_length", "terminal_reason", "achievement_count", *ACHIEVEMENT_NAMES]
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for i, seed in enumerate(seeds):
                writer.writerow(dict(evaluation_seed=int(seed), episode_length=int(arrays["lengths"][i]),
                    terminal_reason=int(arrays["terminal_reason"][i]), achievement_count=int(arrays["achievements"][i].sum()),
                    **dict(zip(ACHIEVEMENT_NAMES, arrays["achievements"][i].astype(int).tolist()))))
        summary = summarize_episodes(arrays["achievements"], ~arrays["active"])
        with np.load(source / "goal_metrics_history.npz") as h:
            rates = h["achievement_success_count"][-458:].sum(0)/h["episode_count"][-458:].sum()
            training_score = float(np.expm1(np.log1p(100*rates).mean()))
        summary.update(job=job, environment=protocol_info, initial_observation_sha256=initial_hash,
            mean_episode_length=float(arrays["lengths"].mean()), elapsed_seconds=time.monotonic()-started,
            training_tail_score22=training_score, score22_minus_training=summary["crafter_score22"]-training_score,
            evaluation_seed_start=manifest["environment_seed_start"], policy_seed=manifest["policy_seed"],
            selector={k: dynamic.get(k, {"score_source":"residual", "structural_prior_weight":1.}.get(k)) for k in (
                "score_source", "structural_prior_weight", "residual_weight", "allow_unready_goals", "allow_completed_goals")},
            termination_counts={name:int((arrays["terminal_reason"]==value).sum()) for value,name in ((1,"health"),(2,"idle"),(3,"time"))},
            total_goal_selections=int(arrays["selected_goal_counts"].sum()))
        for name, expected in job["checkpoint_sha256"].items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
        write_json(directory / "summary.json", summary)
        print(now(), "complete", job["id"], summary["crafter_score22"], flush=True)
    write_json(status_path, dict(updated_at_kst=now(), state="completed", gpu=gpu))


def supervise(root, manifest):
    active = {}
    handles = []
    try:
        for gpu in manifest["gpus"]:
            lock = (Path("/tmp/craftax-oracle-residual-gpu-locks") / f"gpu-{gpu}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(lock)
            log = (root / f"worker-{gpu}.log").open("w")
            env = os.environ.copy();env.pop("LD_LIBRARY_PATH", None)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda,cpu", PYTHONPATH=str(root / "source"),
                XLA_PYTHON_CLIENT_MEM_FRACTION="0.85", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
            process = subprocess.Popen([sys.executable, "-u", str(root / "evaluator-runner.py"),
                "--output", str(root), "--worker", str(gpu)], cwd=root / "source", env=env,
                stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            active[gpu] = (process, log)
        while True:
            completed = [j for j in manifest["jobs"] if (Path(j["output"]) / "summary.json").exists()]
            status = dict(updated_at_kst=now(), completed=len(completed), total=len(manifest["jobs"]), workers=[])
            for gpu, (process, _) in active.items():
                path = root / f"worker-{gpu}.json"
                record = json.loads(path.read_text()) if path.exists() else dict(state="compiling", gpu=gpu)
                record["exit_code"] = process.poll();status["workers"].append(record)
            write_json(root / "status.json", status)
            if all(p.poll() is not None for p, _ in active.values()):
                break
            time.sleep(10)
        if any(p.returncode != 0 for p, _ in active.values()):
            raise RuntimeError("an evaluation worker failed; inspect worker logs")
        assert len(completed) == len(manifest["jobs"])
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
            log.close()
        for lock in handles:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--policy-seed", type=int, default=314159)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0,1,2,3])
    parser.add_argument("--training-sweep", type=Path,
                        help="evaluate every completed checkpoint in this sweep using its frozen source")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--run-prepared", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args();root=args.output.resolve()
    if args.worker is not None:
        worker(root, args.worker);return
    manifest = json.loads((root / "manifest.json").read_text()) if args.run_prepared else prepare(args)
    if not args.dry_run:
        supervise(root, manifest)


if __name__ == "__main__":
    main()
