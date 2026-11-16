#!/usr/bin/env python3
"""Run the exp. 1 vs exp. 2 comparison on a multi-GPU queue.

exp1  : oracle graph + SEA meta-controller, 300M goal steps.
exp2  : SEA meta-controller warmup (50M), then the supervised state-conditioned
        graph + K-hop planner for the rest of the 300M.  K in {2,3,4}, with and
        without the end-node out-degree term ("leaf" / "noleaf").

Jobs are queued in the order exp1 + K=2, then K=3, then K=4 (seeds 0,1,2 each),
and handed to the first free GPU slot.  Each W&B run is grouped by condition so
the project's grouped view shows seed means per condition.

    python scripts/run_dynamic_graph_suite.py --output runs/sea-dynamic3 --dry-run
    python scripts/run_dynamic_graph_suite.py --output runs/sea-dynamic3 --gpus 0 1 2 3

Re-running the same command resumes: finished runs (goal_metrics.json) are
skipped, and unfinished run directories are moved aside and restarted.
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
DEFAULT_GRAPH = REPO / "artifacts/oracle_graphs/craftax_classic_22.npz"

COMMON = [
    "--symbolic",
    "--no-new-tasks",
    "--immortal", "false",
    "--idle-timeout", "10000",
    "--combat-damage-mode", "cumulative",
    "--high-level-goal-timeout", "300",
]


def parse_condition(token):
    """``sea`` or ``K<h>-<leaf|noleaf>-r<repeat>[-p<none|leaf|all>]``.

    e.g. K2-noleaf-r1, K2-leaf-r0-pleaf (prune only the out-degree),
    K2-leaf-r0-pall (also prune every hop by static prerequisites).
    """
    if token == "sea":
        return {"kind": "sea"}
    try:
        parts = token.split("-")
        horizon, leaf, repeat = parts[:3]
        prune = parts[3][1:] if len(parts) == 4 else "none"
        assert len(parts) in (3, 4) and (len(parts) == 3 or parts[3][0] == "p")
        assert horizon[0] == "K" and leaf in ("leaf", "noleaf") and repeat[0] == "r"
        assert prune in ("none", "leaf", "all")
        return {"kind": "planner", "horizon": int(horizon[1:]),
                "leaf": leaf == "leaf", "repeat": float(repeat[1:]), "prune": prune}
    except (ValueError, AssertionError, IndexError):
        raise argparse.ArgumentTypeError(
            f"bad condition {token!r}; use 'sea' or e.g. 'K2-noleaf-r1'"
        ) from None


def default_conditions(args):
    conditions = [parse_condition("sea")]
    for horizon in args.horizons:
        for leaf in args.leaf:
            label = "leaf" if leaf else "noleaf"
            conditions.append(parse_condition(
                f"K{horizon}-{label}-r{args.repeat_target_coefficient:g}"
            ))
    return conditions


def build_jobs(args):
    def job(name, seed, extra, tags):
        return {
            "name": f"{name}-s{seed}",
            "group": name,
            "seed": seed,
            "args": extra,
            "tags": tags,
        }

    conditions = args.conditions or default_conditions(args)
    jobs = []
    for condition in conditions:
        for seed in args.seeds:
            if condition["kind"] == "sea":
                jobs.append(job(
                    "exp1-sea-meta", seed,
                    ["--high-level-controller", "sea-meta-controller"],
                    ["exp1", "sea-meta"],
                ))
                continue
            horizon, repeat = condition["horizon"], condition["repeat"]
            label = "leaf" if condition["leaf"] else "noleaf"
            prune = condition.get("prune", "none")
            suffix = "" if prune == "none" else f"-prune{prune}"
            jobs.append(job(
                f"exp2-K{horizon}-{label}-r{repeat:g}{suffix}", seed,
                [
                    "--high-level-controller", "graph-planner",
                    "--dynamic-warmup-steps", str(args.warmup_steps),
                    "--allow-completed-goals", "true",
                    "--repeat-target-coefficient", str(repeat),
                    "--allow-unready-goals", "true",
                    "--planner-horizon", str(horizon),
                    "--planner-leaf", str(condition["leaf"]).lower(),
                    "--high-level-q-learning-rate", str(args.model_learning_rate),
                    "--planner-prune", prune,
                ],
                ["exp2", f"K{horizon}", label, f"repeat{repeat:g}", f"prune-{prune}"],
            ))
    if args.order == "seed":
        # Every condition's seed 0 first, then seed 1, then seed 2.
        jobs.sort(key=lambda item: args.seeds.index(item["seed"]))
    return jobs


def command(args, job, output):
    cmd = [
        args.python, "-m", "craftax.sea.train",
        *COMMON,
        "--graph-path", str(args.graph),
        "--goal-timesteps", str(args.goal_timesteps),
        "--seed", str(job["seed"]),
        "--goal-success-coefficient", str(args.goal_success_coefficient),
        "--log-interval", str(args.log_interval),
        "--output", str(output),
        *job["args"],
    ]
    if args.wandb_mode != "disabled":
        cmd += [
            "--wandb-project", args.wandb_project,
            "--wandb-group", job["group"],
            "--wandb-name", job["name"],
            "--wandb-mode", args.wandb_mode,
            "--wandb-tags", *job["tags"], f"seed{job['seed']}",
        ]
        if args.wandb_entity:
            cmd += ["--wandb-entity", args.wandb_entity]
    return cmd


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=1,
                        help="concurrent runs per GPU; memory is split evenly")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizons", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--leaf", nargs="+", type=lambda v: v.lower() == "true",
                        default=[True, False], metavar="BOOL")
    parser.add_argument(
        "--order", choices=("horizon", "seed"), default="horizon",
        help="queue order: by K (all seeds per K) or by seed (all conditions per seed)",
    )
    parser.add_argument("--goal-timesteps", type=int, default=300_000_000)
    parser.add_argument("--warmup-steps", type=int, default=50_000_000)
    parser.add_argument("--model-learning-rate", type=float, default=3e-4)
    parser.add_argument("--repeat-target-coefficient", type=float, default=0.0,
                        help="repeat reward used by the default K x leaf grid")
    parser.add_argument("--goal-success-coefficient", type=float, default=3.0,
                        help="first-time target reward for every run (SEA original: 5)")
    parser.add_argument("--conditions", nargs="+", type=parse_condition,
                        help="explicit conditions instead of the grid, e.g. sea K2-leaf-r0 K2-noleaf-r1"),
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jax-platforms", default="cuda,cpu",
                        help="JAX_PLATFORMS for children (cpu is needed for progress callbacks)")
    parser.add_argument("--skip-gpu-check", action="store_true",
                        help="do not verify that every GPU is visible to JAX before starting")
    parser.add_argument("--wandb-project", default="sea-dynamic3")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    parser.add_argument("--label", default="",
                        help="suffix for manifest/status/supervisor files when several supervisors share --output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be positive")
    args.graph = args.graph.expanduser().resolve()

    root = Path(args.output).expanduser().resolve()
    jobs = build_jobs(args)
    for job in jobs:
        job["output"] = str(root / job["group"] / f"seed-{job['seed']}")
        job["command"] = command(args, job, job["output"])

    if args.dry_run:
        for job in jobs:
            print(f"# {job['name']}\n{shlex.join(job['command'])}\n")
        print(f"{len(jobs)} jobs, {len(args.gpus) * args.jobs_per_gpu} concurrent slots")
        return

    if not args.skip_gpu_check:
        for gpu in args.gpus:
            check = subprocess.run(
                [args.python, "-c", "import jax; print(jax.default_backend())"],
                cwd=REPO, capture_output=True, text=True,
                env={**{k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"},
                     "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
            backend = check.stdout.strip().splitlines()[-1:] or ["?"]
            if backend[0] != "gpu":
                sys.exit(f"GPU {gpu} is not usable by JAX (backend={backend[0]}):\n{check.stderr[-2000:]}")

    root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"manifest{args.label}.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "argv": sys.argv,
        "jobs": jobs,
    })
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
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
        write_json(root / f"status{args.label}.json", {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "supervisor_pid": os.getpid(),
            "jobs": [
                {k: job.get(k) for k in ("name", "state", "gpu", "pid", "returncode", "output")}
                for job in jobs
            ],
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
                "JAX_PLATFORMS": args.jax_platforms,
            })
            process = subprocess.Popen(
                job["command"], cwd=REPO, env=environment,
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
        print("failed: " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
