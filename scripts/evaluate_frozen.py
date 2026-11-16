#!/usr/bin/env python3
"""Evaluate finished runs with frozen checkpoints on held-out seeds.

    python scripts/evaluate_frozen.py runs/sea-dynamic3/exp1-sea-meta/seed-0 ...

Defaults: seeds 84001-84005, 256 environments, about 1M steps per seed.
Writes ``eval_frozen.json`` next to each run's checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from craftax.sea.frozen_eval import DEFAULT_SEEDS, evaluate_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories (containing config.json)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps-per-seed", type=int, default=1_000_000)
    parser.add_argument("--output-name", default="eval_frozen.json")
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        result = evaluate_run(run, seeds=args.seeds, num_envs=args.num_envs,
                              steps_per_seed=args.steps_per_seed)
        (Path(run) / args.output_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
        mean = result["mean_over_seeds"]
        pooled = result["pooled"]
        label = f"{Path(run).parent.name}/{Path(run).name}"
        rows.append((label, mean, pooled))
        print(f"{label}: Score22 {mean['crafter_score22']:.2f} ± {mean['crafter_score22_std']:.2f} | "
              f"Score21 {mean['crafter_score21']:.2f} ± {mean['crafter_score21_std']:.2f} | "
              f"episodes {pooled['episodes']} | steps/seed {result['steps_per_seed']}", flush=True)
        for seed in result["per_seed"]:
            print(f"    seed {seed['seed']}: Score22 {seed['crafter_score22']:.2f} "
                  f"Score21 {seed['crafter_score21']:.2f} episodes {seed['episodes']} "
                  f"len {seed['mean_episode_length']:.0f}", flush=True)


if __name__ == "__main__":
    main()
