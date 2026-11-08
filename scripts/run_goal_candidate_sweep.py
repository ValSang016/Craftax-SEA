"""Prepare or run frozen oracle goal-selection sweeps on a four-GPU queue."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = (("ready-repeat", False, True), ("all-repeat", True, True), ("all-once", True, False))
SELECTOR_ABLATIONS = (
    ("residual-no-prior", 0.8, 0.0, "residual"),
    ("residual-no-prior", 10.0, 0.0, "residual"),
    ("base-q-with-prior", 0.5, 1.0, "base-q"),
    ("base-q-with-prior", 0.3, 1.0, "base-q"),
    ("base-q-with-prior", 0.2, 1.0, "base-q"),
)


def experiment_settings(suite):
    if suite == "full-q":
        return [dict(condition="full-q-with-prior", beta=beta, structural_prior_weight=1.0,
                     score_source="full-q", allow_unready_goals=True, allow_completed_goals=False)
                for beta in (0.2, 0.3, 0.5)]
    if suite == "selector-ablations":
        return [dict(condition=condition, beta=beta, structural_prior_weight=weight,
                     score_source=signal, allow_unready_goals=True, allow_completed_goals=False)
                for condition, beta, weight, signal in SELECTOR_ABLATIONS]
    if suite == "goal-candidates":
        return [dict(condition=condition, beta=beta, structural_prior_weight=1.0,
                     score_source="residual", allow_unready_goals=unready, allow_completed_goals=completed)
                for beta in (0.0, 0.8, 10.0) for condition, unready, completed in CONDITIONS]
    raise ValueError(f"unknown suite: {suite}")


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_plan(args):
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing sweep: {root}")
    reference = json.loads(args.reference_config.read_text())
    root.mkdir(parents=True)
    source = root / "source"
    shutil.copytree(REPO / "craftax", source / "craftax", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, root / "runner-snapshot.py")
    shutil.copy2(args.reference_config, root / "reference-config.json")
    shutil.copy2(args.graph, root / "oracle_graph.npz")
    (root / "source.patch").write_text(subprocess.check_output(["git", "diff", "--", "craftax", "tests", "docs"], cwd=REPO, text=True))
    saved = reference["args"]
    settings = (
        "goal_timesteps", "num_envs", "num_steps", "update_epochs", "num_minibatches",
        "learning_rate", "gamma", "gae_lambda", "clip_epsilon", "entropy_coefficient",
        "value_coefficient", "max_grad_norm", "hidden_size", "reset_ratio", "idle_timeout",
        "health_reward_coef", "combat_damage_mode", "contrast_weight", "goal_success_coefficient",
        "non_target_achievement_coefficient", "dynamic_warmup_steps", "dynamic_temperature",
        "high_level_replay_capacity", "high_level_q_batch_size", "high_level_q_learning_rate",
        "high_level_q_replay_ratio", "high_level_min_replay_size", "high_level_goal_timeout",
    )
    command = [str(args.python), "-u", "-m", "craftax.sea.train", "--symbolic",
               "--high-level-controller", "residual", "--graph-path", str(root / "oracle_graph.npz"),
               "--immortal", str(saved["immortal"]).lower(), "--log-interval", "10"]
    if saved["no_new_tasks"]:
        command += ["--no-new-tasks"]
    for name in settings:
        command += ["--" + name.replace("_", "-"), str(saved[name])]
    command += ["--wandb-mode", "offline", "--wandb-project", saved["wandb_project"], "--wandb-entity", saved["wandb_entity"]]
    jobs = []
    for seed in (0, 1, 2):
        for setting in experiment_settings(args.suite):
            condition, beta = setting["condition"], setting["beta"]
            directory = root / condition / f"seed-{seed}" / ("beta-" + str(beta).replace(".", "p"))
            name = f"{root.name}-{condition}-s{seed}-b{beta:g}"
            argv = command + ["--seed", str(seed), "--dynamic-beta", str(beta),
                "--dynamic-lambda", str(setting["structural_prior_weight"]),
                "--dynamic-score-source", setting["score_source"],
                "--allow-unready-goals", str(setting["allow_unready_goals"]).lower(),
                "--allow-completed-goals", str(setting["allow_completed_goals"]).lower(),
                "--wandb-name", name, "--wandb-group", root.name, "--output", str(directory)]
            jobs.append(dict(id=len(jobs), seed=seed, **setting,
                output=str(directory), command=argv, state="queued"))
    manifest = dict(created_at_kst=now(), reference=str(args.reference_config.resolve()),
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        source_snapshot=str(source), source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob("*.py"))},
        graph_sha256=hashlib.sha256((root / "oracle_graph.npz").read_bytes()).hexdigest(),
        suite=args.suite, gpu_ids=args.gpus, num_runs=len(jobs), timesteps_per_run=saved["goal_timesteps"],
        warmup_steps=saved["dynamic_warmup_steps"], wandb_mode="offline", jobs=jobs,
        repeat_success="actual repeated event ends the option; no repeat achievement/target bonus",
        achievement_reward_policy="episode-first-occurrence-only in environment, high-level Q, and worker",
        environment_reward_and_scores="unique first achievements plus existing health shaping; unchanged",
        warmup_policy="ready only; completed-goal eligibility follows its flag",
        empty_candidate_fallback="legacy empty-set fallback retained, including all goals if all nodes are completed")
    write_json(root / "manifest.json", manifest)
    return manifest


def load_frozen_plan(root):
    """Validate an existing plan without changing source or output artifacts."""
    manifest = json.loads((root / "manifest.json").read_text())
    source = Path(manifest["source_snapshot"])
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen training source changed: {relative}")
    if hashlib.sha256((root / "oracle_graph.npz").read_bytes()).hexdigest() != manifest["graph_sha256"]:
        raise ValueError("frozen graph changed")
    for job in manifest["jobs"]:
        directory = Path(job["output"])
        if not directory.is_relative_to(root):
            raise ValueError(f"experiment output outside sweep: {directory}")
        if job["state"] == "completed":
            for filename in ("goal_metrics.json", "goal_metrics_history.npz"):
                if not (directory / filename).exists():
                    raise ValueError(f"completed experiment missing {filename}: {directory}")
    return manifest


def load_resume_plan(root):
    manifest = load_frozen_plan(root)
    if any(job["state"] not in {"completed", "failed"} for job in manifest["jobs"]):
        raise ValueError("failed-only retry requires a settled sweep")
    if not any(job["state"] == "failed" for job in manifest["jobs"]):
        raise ValueError("no failed experiments to retry")
    return manifest


def load_prepared_plan(root):
    manifest = load_frozen_plan(root)
    if not all(job["state"] == "queued" for job in manifest["jobs"]):
        raise ValueError("prepared sweep must contain only unstarted queued jobs")
    if any(Path(job["output"]).exists() for job in manifest["jobs"]):
        raise FileExistsError("prepared experiment output already exists")
    return manifest


def requeue_failed_jobs(root, manifest):
    """Archive previous failed attempts without touching completed experiments."""
    archive = root / "retry-attempts" / datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S-%f")
    archive.mkdir(parents=True)
    for filename in ("manifest.json", "status.json"):
        if (root / filename).exists():
            shutil.copy2(root / filename, archive / filename)
    shutil.copy2(__file__, archive / "runner-snapshot.py")
    retried = []
    for job in manifest["jobs"]:
        if job["state"] != "failed":
            continue
        directory = Path(job["output"])
        previous = {key: job.pop(key) for key in (
            "gpu", "pid", "started_at_kst", "finished_at_kst", "exit_code"
        ) if key in job}
        archived_output = archive / directory.relative_to(root)
        if directory.exists():
            archived_output.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(archived_output)
        previous.update(state="failed", archived_output=str(archived_output))
        job.setdefault("previous_attempts", []).append(previous)
        job["state"] = "queued"
        retried.append(job["id"])
    manifest.setdefault("retry_history", []).append(dict(
        resumed_at_kst=now(), job_ids=retried, archive=str(archive),
        policy="rerun failed jobs from step zero with original frozen source; preserve completed jobs",
    ))
    write_json(root / "manifest.json", manifest)


def summarize(root, python, source):
    code = '''
import csv,json,sys
from pathlib import Path
import numpy as np
from craftax.sea.metrics import EASY_ACHIEVEMENT_INDICES,HARD_ACHIEVEMENT_INDICES
root=Path(sys.argv[1]);jobs=json.loads((root/'manifest.json').read_text())['jobs'];rows=[]
for job in jobs:
 if job['state']!='completed':continue
 p=Path(job['output'])/'goal_metrics_history.npz'
 if not p.exists():continue
 with np.load(p) as h:
  n=len(h['episode_count']);w=min(458,n);count=h['achievement_success_count'][-w:].sum(0);ep=h['episode_count'][-w:].sum();r=count/max(ep,1)
  rows.append(dict(condition=job['condition'],beta=job['beta'],structural_prior_weight=job.get('structural_prior_weight',1.0),score_source=job.get('score_source','residual'),seed=job['seed'],updates=n,window_updates=w,episodes=int(ep),crafter_score=float(np.expm1(np.log1p(100*r).mean())),achievements_per_episode=float(r.sum()),easy_success_rate=float(r[list(EASY_ACHIEVEMENT_INDICES)].mean()),hard_success_rate=float(r[list(HARD_ACHIEVEMENT_INDICES)].mean()),mean_reward=float(h['mean_reward'][-w:].mean())))
if rows:
 with (root/'final-window.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 (root/'final-window.json').write_text(json.dumps(rows,indent=2))
 groups=[]
 for condition in sorted({r['condition'] for r in rows}):
  for beta in sorted({r['beta'] for r in rows}):
   selected=[r for r in rows if r['condition']==condition and r['beta']==beta]
   if not selected:continue
   g=dict(condition=condition,beta=beta,structural_prior_weight=selected[0]['structural_prior_weight'],score_source=selected[0]['score_source'],num_seeds=len(selected))
   for metric in ['crafter_score','achievements_per_episode','easy_success_rate','hard_success_rate','mean_reward']:
    values=np.asarray([r[metric] for r in selected]);g[metric+'_mean']=float(values.mean());g[metric+'_seed_std']=float(values.std(ddof=1)) if len(values)>1 else None
   groups.append(g)
 with (root/'final-window-by-condition.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=groups[0]);writer.writeheader();writer.writerows(groups)
'''
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1")
    subprocess.run([str(python), "-c", code, str(root)], cwd=source, env=env, check=True)
    analysis = root / "beta-statistics-snapshot.py"
    if analysis.exists():
        subprocess.run([str(python), str(analysis), str(root)], cwd=source, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="physical GPU IDs; default: saved plan GPUs, or 0 1 2 3 for a new plan")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--reference-config", type=Path, default=REPO / "runs/oracle-residual/seed-0/beta-0p8/config.json")
    parser.add_argument("--graph", type=Path, default=REPO / "artifacts/oracle_graphs/craftax_classic_22.npz")
    parser.add_argument("--suite", choices=("goal-candidates", "selector-ablations", "full-q"), default="goal-candidates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Explicitly start a selector ablation suite; otherwise it only prepares a frozen plan")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume-failed", action="store_true", help="Retry only failed jobs in an existing sweep, preserving completed results and the frozen training source")
    existing.add_argument("--start-prepared", action="store_true", help="Start an existing dry-run plan using its frozen source and exact commands")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.gpus is None:
        args.gpus = (json.loads((root / "manifest.json").read_text())["gpu_ids"]
                     if args.resume_failed or args.start_prepared else [0, 1, 2, 3])
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be nonempty and unique")
    if args.resume_failed:
        manifest = load_resume_plan(root)
    elif args.start_prepared:
        manifest = load_prepared_plan(root)
    else:
        manifest = make_plan(args)
    source = Path(manifest["source_snapshot"])
    jobs = manifest["jobs"]
    suite = manifest.get("suite", "goal-candidates")
    if args.dry_run or (suite in ("selector-ablations", "full-q") and not args.execute):
        count = sum(job["state"] == "failed" for job in jobs) if args.resume_failed else len(jobs)
        print(f"Validated {count} commands in {root / 'manifest.json'}; training not started", flush=True)
        return
    locks = {}
    active = {}
    stopped = [False]
    gpu_available = True
    lock_root = Path("/tmp/craftax-oracle-residual-gpu-locks")
    lock_root.mkdir(exist_ok=True)
    def stop(signum, frame):
        stopped[0] = True
        for process, _, _ in active.values():
            process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for gpu in args.gpus:
            handle = (lock_root / f"gpu-{gpu}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise RuntimeError(f"GPU {gpu} is reserved by another sweep")
            locks[gpu] = handle
            (lock_root / f"gpu-{gpu}.owner").write_text(f"{os.getpid()} {root} {now()}\n")
        if args.resume_failed:
            requeue_failed_jobs(root, manifest)
        analysis = REPO / "scripts/analyze_goal_candidate_sweep.py"
        if analysis.exists() and suite == "goal-candidates":
            shutil.copy2(analysis, root / "beta-statistics-snapshot.py")
        manifest["gpu_ids"] = args.gpus
        summarize(root, args.python, source)
        while True:
            newly_finished = False
            for gpu, (process, job, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                complete = code == 0 and (Path(job["output"]) / "goal_metrics.json").exists()
                job.update(state="completed" if complete else "failed", exit_code=code, finished_at_kst=now())
                print(now(), job["state"], job["id"], job["condition"], job["beta"], job["seed"], flush=True)
                del active[gpu]
                newly_finished = True
            if newly_finished:
                write_json(root / "manifest.json", manifest)
                summarize(root, args.python, source)
            if not stopped[0] and len(active) < len(args.gpus) and any(j["state"] == "queued" for j in jobs):
                available = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=20, check=False,
                ).returncode == 0
                if available != gpu_available:
                    print(now(), "GPU recovered" if available else "GPU unavailable; preserving queued jobs until recovery", flush=True)
                gpu_available = available
            if not stopped[0] and gpu_available:
                for gpu in args.gpus:
                    if gpu in active:
                        continue
                    job = next((j for j in jobs if j["state"] == "queued"), None)
                    if job is None:
                        break
                    directory = Path(job["output"])
                    directory.mkdir(parents=True)
                    log = (directory / "train.log").open("w")
                    env = os.environ.copy()
                    env.pop("LD_LIBRARY_PATH", None)
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda,cpu", PYTHONPATH=str(source),
                        XLA_PYTHON_CLIENT_MEM_FRACTION="0.90", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                    process = subprocess.Popen(job["command"], cwd=source, env=env, stdout=log,
                        stderr=subprocess.STDOUT, pass_fds=(locks[gpu].fileno(),))
                    job.update(state="running", gpu=gpu, pid=process.pid, started_at_kst=now())
                    active[gpu] = (process, job, log)
                    print(now(), "started", job["id"], job["condition"], job["beta"], job["seed"], "gpu", gpu, flush=True)
            status = dict(updated_at_kst=now(), scheduler_pid=os.getpid(),
                gpu_available=gpu_available,
                completed=sum(j["state"] == "completed" for j in jobs), failed=sum(j["state"] == "failed" for j in jobs),
                queued=sum(j["state"] == "queued" for j in jobs), running=[])
            for gpu, (_, job, _) in active.items():
                item = {k:job[k] for k in ("id", "condition", "beta", "seed", "gpu", "pid", "output")}
                progress = Path(job["output"]) / "progress.json"
                if progress.exists():
                    record = json.loads(progress.read_text())
                    item.update(update=record["update"], steps=record["steps"], total_updates=record["total_updates"])
                status["running"].append(item)
            write_json(root / "manifest.json", manifest)
            write_json(root / "status.json", status)
            if not active and (stopped[0] or status["queued"] == 0):
                break
            time.sleep(10)
        if not stopped[0]:
            summarize(root, args.python, source)
        print(now(), "finished", json.dumps(status), flush=True)
    finally:
        for gpu, handle in locks.items():
            (lock_root / f"gpu-{gpu}.owner").unlink(missing_ok=True)
            handle.close()
    if stopped[0]:
        raise SystemExit(130)
    if any(job["state"] == "failed" for job in jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
