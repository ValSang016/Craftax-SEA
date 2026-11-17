#!/usr/bin/env python3
"""Queue graph-planner runs over GPUs.

Conditions are ``K<h>-<leaf|noleaf>-r<x>[-p<none|leaf|all>][-nore][-e<eps>]``:
K hops, leaf term on/off, repeat-target reward x, optional static prune, and
``nore`` = completed goals cannot be selected again, ``e<eps>`` = after
warmup a uniform random candidate goal with probability eps.  Jobs are queued seed by
seed (every condition's seed 0, then seed 1, ...) and handed to the first free
GPU slot.

    python scripts/run_planner_suite.py --output runs/planner --group planner-v1 --dry-run
    python scripts/run_planner_suite.py --output runs/planner --group planner-v1 \
        --gpus 0 --jobs-per-gpu 3

Re-running resumes: finished runs (goal_metrics.json) are skipped and
unfinished run directories are moved aside and restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONDITIONS = ("K2-leaf-r0", "K2-leaf-r0-nore", "K2-noleaf-r0")


def parse_condition(token):
    try:
        horizon, leaf, repeat, *extras = token.split("-")
        if horizon[0] != "K" or leaf not in ("leaf", "noleaf") or repeat[0] != "r":
            raise ValueError
        condition = {
            "name": token,
            "horizon": int(horizon[1:]),
            "leaf": leaf == "leaf",
            "repeat": float(repeat[1:]),
            "prune": "none",
            "reselect": True,
            "epsilon": 0.0,
        }
        for extra in extras:
            if extra == "nore":
                condition["reselect"] = False
            elif extra[:1] == "e" and extra[1:]:
                condition["epsilon"] = float(extra[1:])
                if not 0 <= condition["epsilon"] <= 1:
                    raise ValueError
            elif extra[:1] == "p" and extra[1:] in ("none", "leaf", "all"):
                condition["prune"] = extra[1:]
            else:
                raise ValueError
        return condition
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"bad condition {token!r}; e.g. K2-leaf-r0, K2-leaf-r0-nore-e0.2, K3-noleaf-r1-pleaf"
        ) from None


def condition_args(condition):
    return [
        "--planner-horizon", str(condition["horizon"]),
        "--planner-leaf", str(condition["leaf"]).lower(),
        "--repeat-target-coefficient", f"{condition['repeat']:g}",
        "--planner-prune", condition["prune"],
        "--allow-completed-goals", str(condition["reselect"]).lower(),
        "--planner-epsilon", f"{condition['epsilon']:g}",
    ]


def build_jobs(args):
    jobs = []
    for seed in args.seeds:
        for condition in args.conditions:
            output = Path(args.output).resolve() / condition["name"] / f"seed-{seed}"
            command = [
                args.python, "-m", "craftax.sea_planner.train",
                "--output", str(output),
                "--seed", str(seed),
                "--goal-timesteps", str(args.goal_timesteps),
                "--dynamic-warmup-steps", str(args.warmup_steps),
                "--log-interval", str(args.log_interval),
                *condition_args(condition),
                *args.extra,
            ]
            if args.wandb_mode != "disabled":
                command += [
                    "--wandb-project", args.wandb_project,
                    "--wandb-group", args.group,
                    "--wandb-job-type", condition["name"],
                    "--wandb-name", f"{condition['name']}-s{seed}",
                    "--wandb-mode", args.wandb_mode,
                    "--wandb-tags", condition["name"], f"seed{seed}",
                ]
                if args.wandb_entity:
                    command += ["--wandb-entity", args.wandb_entity]
            jobs.append({"name": f"{condition['name']}-s{seed}", "seed": seed,
                         "output": str(output), "command": command})
    return jobs


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group", required=True, help="W&B group for every run")
    parser.add_argument("--conditions", nargs="+", type=parse_condition,
                        default=[parse_condition(c) for c in DEFAULT_CONDITIONS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--goal-timesteps", type=int, default=300_000_000)
    parser.add_argument("--warmup-steps", type=int, default=50_000_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb-project", default="craftax-sea-planner")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "zaezae01"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="further craftax.sea_planner.train options (must be last)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be positive")

    jobs = build_jobs(args)
    if args.dry_run:
        for job in jobs:
            print(f"# {job['name']}\n{shlex.join(job['command'])}\n")
        print(f"{len(jobs)} jobs, {len(args.gpus) * args.jobs_per_gpu} concurrent slots")
        return

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "jobs": jobs,
    })
    pending = []
    for job in jobs:
        output = Path(job["output"])
        if (output / "goal_metrics.json").is_file():
            job["state"] = "done"
            continue
        if output.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output.rename(output.with_name(f"{output.name}.incomplete-{stamp}"))
        job["state"] = "queued"
        pending.append(job)

    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    running = {}
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True
        for process, _, _ in running.values():
            process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def status():
        write_json(root / "status.json", {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "supervisor_pid": os.getpid(),
            "jobs": [{k: job.get(k) for k in ("name", "state", "gpu", "pid", "returncode", "output")}
                     for job in jobs],
        })

    status()
    while (pending or running) and not stopping:
        free = [slot for slot in range(len(slots)) if slot not in running]
        while pending and free:
            slot = free.pop(0)
            job = pending.pop(0)
            output = Path(job["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            log = (output.parent / f"seed-{job['seed']}.log").open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment.pop("LD_LIBRARY_PATH", None)
            environment.update({
                "CUDA_VISIBLE_DEVICES": str(slots[slot]),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{0.9 / args.jobs_per_gpu:.2f}",
                "JAX_PLATFORMS": "cuda,cpu",
                "PYTHONUNBUFFERED": "1",
            })
            process = subprocess.Popen(
                job["command"], cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            job.update(state="running", gpu=slots[slot], pid=process.pid)
            running[slot] = (process, job, log)
            print(f"[start] {job['name']} gpu={slots[slot]} pid={process.pid}", flush=True)
            status()
        time.sleep(10)
        for slot, (process, job, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[slot]
            job.update(state="done" if code == 0 else "failed", returncode=code)
            print(f"[{job['state']}] {job['name']} returncode={code}", flush=True)
            status()

    for process, job, log in running.values():
        process.wait()
        log.close()
        job.update(state="stopped", returncode=process.returncode)
    status()
    failed = [job["name"] for job in jobs if job.get("state") == "failed"]
    if failed:
        sys.exit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
