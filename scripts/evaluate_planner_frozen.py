#!/usr/bin/env python3
"""Frozen held-out evaluation (Score22 / Score21 mean ± std over env seeds).

    python scripts/evaluate_planner_frozen.py runs/planner/K2-leaf-r0/seed-0 \
        runs/oracle22-mortal-org-unk0/seed-0

Works for graph-planner runs and SEA oracle-graph baseline runs.  Defaults:
env seeds 84001-84005, 256 envs x 62 updates x 63 steps (~1M steps) per seed.
Writes ``eval_frozen.json`` into each run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from craftax.sea_planner.frozen_eval import DEFAULT_SEEDS, evaluate_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps-per-seed", type=int, default=1_000_000)
    parser.add_argument("--output-name", default="eval_frozen.json")
    args = parser.parse_args()
    for run in args.runs:
        result = evaluate_run(run, seeds=args.seeds, num_envs=args.num_envs,
                              steps_per_seed=args.steps_per_seed)
        (Path(run) / args.output_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
        mean = result["mean_over_seeds"]
        print(f"{run} [{result['controller']}]: "
              f"Score22 {mean['crafter_score22']:.2f} ± {mean['crafter_score22_std']:.2f} | "
              f"Score21 {mean['crafter_score21']:.2f} ± {mean['crafter_score21_std']:.2f} | "
              f"episodes {result['pooled']['episodes']}", flush=True)


if __name__ == "__main__":
    main()
