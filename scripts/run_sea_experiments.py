"""Run independent SEA seeds on dedicated GPUs and retain execution records."""

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-run-root", help="legacy option; rejected by streaming discovery")
    parser.add_argument("--base-run-root", help="directory with existing seed-N/base_policy.msgpack checkpoints")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != len(args.seeds) or len(set(args.gpus)) != len(args.gpus):
        parser.error("provide one distinct GPU per seed")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")

    if args.discovery_run_root:
        parser.error("--discovery-run-root is incompatible with fresh streaming discovery; reuse --base-run-root only")

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        parser.error("output already contains a run; choose a new directory")
    source = output / "source"
    source.mkdir()
    shutil.copytree(root / "craftax", source / "craftax",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("SEA_README.md", "requirements-sea-cuda.txt", "environment-sea.yml"):
        shutil.copy2(root / name, source / name)
    for name, command in (
        ("pip-freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
        ("source.diff", ["git", "-c", f"safe.directory={root}", "diff", "HEAD"]),
        ("source-commit.txt", ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"]),
        ("gpu-start.txt", ["nvidia-smi"]),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                           check=True)

    state = {"supervisor_pid": os.getpid(), "started_at": time.time(),
             "state": "starting", "python": sys.executable, "runs": []}
    children = []

    def stop_children():
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for gpu, seed in zip(args.gpus, args.seeds):
            run_output = output / f"seed-{seed}"
            run_output.mkdir()
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "JAX_PLATFORMS": "cuda,cpu",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                        "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
            env.pop("LD_LIBRARY_PATH", None)
            env.pop("PYTHONPATH", None)
            command = [sys.executable, "-u", "-m", "craftax.sea.train", "--pixels",
                       "--seed", str(seed), "--log-interval", "10",
                       "--output", str(run_output)]
            if args.base_only:
                command += ["--base-only"]
            if args.base_run_root:
                checkpoint = Path(args.base_run_root).resolve() / f"seed-{seed}" / "base_policy.msgpack"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                command += ["--base-policy-checkpoint", str(checkpoint)]
            with (run_output / "train.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=source, env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT)
            children.append(child)
            state["runs"].append({"gpu": gpu, "seed": seed, "pid": child.pid,
                                  "command": command, "output": str(run_output),
                                  "returncode": None})
            print(f"started seed={seed} GPU={gpu} pid={child.pid}", flush=True)
        state["state"] = "running"
        while True:
            for child, record in zip(children, state["runs"]):
                record["returncode"] = child.poll()
            state["updated_at"] = time.time()
            write_json(output / "status.json", state)
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("a training process failed; see its train.log")
            if all(child.poll() == 0 for child in children):
                state["state"] = "completed"
                results = {}
                for record in state["runs"]:
                    folder = Path(record["output"])
                    results[str(record["seed"])] = {
                        stage: json.loads((folder / f"{stage}_metrics.json").read_text())
                        for stage in (("base",) if args.base_only else ("base", "goal"))
                    }
                write_json(output / "results.json", results)
                break
            time.sleep(10)
    except BaseException as error:
        state["state"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = str(error)
        stop_children()
        raise
    finally:
        for child, record in zip(children, state["runs"]):
            record["returncode"] = child.poll()
        state["updated_at"] = time.time()
        write_json(output / "status.json", state)


if __name__ == "__main__":
    main()
