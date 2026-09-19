# SEA on Craftax-Classic with PPO-RNN

**기본 버전: SEA (fix123).** [코드·실험 안내](docs/sea/README.md) ·
[원본과의 차이](docs/sea/alignment.md)

fix123는 현재 기준 포팅이며, 원본 ICLR-SEA와 수치적으로 동일한 reproduction은 아닙니다.

This implementation keeps the original Craftax-Classic environments intact and
adds SEA-specific modules under `craftax/sea`.

## Environment semantics

- player health is always 9 (immortal);
- lava and health never terminate an episode;
- 100 consecutive steps without a new achievement terminate the episode;
- the 10,000-step hard limit remains;
- reward is newly unlocked achievement count, with no health shaping;
- creature damage is lethal-only and cannot accumulate; and
- cow, skeleton and zombie health are 2, 3 and 5.

## Quick smoke run

From the `Craftax-main` directory:

```powershell
python -m craftax.sea.train `
  --symbolic-debug `
  --base-timesteps 4096 `
  --goal-timesteps 4096 `
  --discovery-interactions 8192 `
  --discovery-batch-size 256 `
  --discovery-num-envs 16 `
  --clustering-transitions 32 `
  --clustering-interactions 65536 `
  --num-envs 16 `
  --num-steps 16 `
  --num-minibatches 4 `
  --hidden-size 64 `
  --reset-ratio 4 `
  --min-clusters 2 `
  --max-clusters 4 `
  --output runs/smoke
```

Raw pixels are the default and define the primary experiment.  The quick command
above deliberately adds `--symbolic-debug` only to debug JAX shapes and
termination semantics quickly.  Omitting it runs the pixel experiment;
`--pixels` is also accepted as an explicit declaration.

The encoder now learns online from **fresh** frozen-policy rollouts. The default
32 collection environments x 80 steps provide 2,560 transitions per update.
Every transition, including zero-reward ones, is used once for reward-occurrence
prediction. There is no positive oversampling and no repeated 200k replay dataset.
A 50M-interaction budget therefore trains on exactly 50M prediction examples in
19,532 updates; the final batch contains 640 transitions. Only one rollout batch
and the bounded episode buffer need to be retained.

During the first 1M interactions, all achievement transitions are accumulated by
episode. Completed episodes with at least two events enter a FIFO of the latest
256 episodes. Contrast learning begins after six eligible episodes complete,
selecting 128 episode groups and at most eight events per group. At the 1M cutoff,
unfinished episodes are discarded and the FIFO freezes. Prediction learning keeps
using new rollout transitions for the rest of the budget, while contrast learning
continues sampling the frozen complete-episode buffer.

After encoder training finishes, an independent rollout (seed + 4) supplies the
first 10,000 positive transitions for clustering. Use `--clustering-transitions`
and `--clustering-interactions` to set the event count and maximum rollout budget.
Insufficient positive events cause an explicit error rather than silent underfill.
These transitions are saved in `clustering_dataset.npz`; they are not reused from
encoder training. `discovery_summary.json` records actual prediction example
counts, their natural positive fraction, optimizer updates and contrast groups.
The final complete-episode buffer is saved as `contrast_episode_dataset.npz` when
nonempty. Full prediction frames are streamed and not saved to a replay archive.

`--discovery-batch-size` sets fresh examples per update and must be divisible by
`--discovery-num-envs`. Update count is derived from the interaction budget. A
conflicting `--discovery-train-steps` is rejected to avoid silently dropping or
repeating data. `--discovery-dataset-checkpoint` and the experiment supervisor's
`--discovery-run-root` are rejected for streaming training. Saved PPO collection
checkpoints remain supported via `--base-policy-checkpoint` / `--base-run-root`.
The old offline collector and learner functions remain available for diagnostics,
but the primary CLI does not use them.

## Alignment with original SEA

- policy, recurrent, transition, and clustering embedding widths default to 256;
- the pixel torso uses SEA's 8/4, 4/2, 3/1 convolutions and two 256-wide layers;
- objectives are one-hot encoded by two 256-wide layers, without the legacy
  completed-objective vector;
- representation learning uses Adam at 1e-4, 128 contrast episode groups, at
  most 8 events per episode, a 256-episode contrast buffer, and contrast weight
  20;
- collection policy, discovery, and goal-policy budgets are 200M, 50M, and
  300M environment interactions respectively.

The policy learner intentionally remains Craftax's JAX PPO-RNN design: 1024
parallel environments x 64 rollout steps, GAE, clipped PPO, Adam, GRU, four
epochs, and eight recurrent minibatches.  The entropy coefficient is aligned
to SEA at 0.001.  Original SEA uses asynchronous IMPALA/V-trace, RMSProp, LSTM,
summed losses, and a gradient clip threshold of 40.  Copying the numeric clip
value 40 onto mean-reduced PPO losses would almost disable clipping; 1.0 is
therefore retained and exposed as `--max-grad-norm`.
Discovery losses use the original reference **relative** weighting:
`mean(prediction_loss) + 1 * mean(contrast_loss)`. The coefficient is derived from
`20 * 128 / 2560 = 1`, using the original SEA reference counts. It does not change
when a smaller working batch or contrast-group count is requested. This fixes
the previous port's effective multiplier of 10. Loss scale, optimizer epsilon,
framework initialization and PPO/IMPALA differences are not claimed identical.

## Metrics

Each PPO stage writes `base_metrics.json` or `goal_metrics.json`.  The summary
contains completed episode count, mean return and length, every achievement's
success count/rate, easy/hard mean success rates, and the standard Crafter score
`exp(mean(log(1 + 100 * success_rate))) - 1`.  Rates use completed episodes as
their denominator; rollout steps and goal pseudo-terminals are not counted as
episodes.  PPO diagnostics also include actor/value loss, entropy, pre-clip
gradient norm, and the fraction of minibatches whose norm exceeded the clip.
The corresponding `*_metrics_history.npz` files retain every PPO update rather
than only the final/cumulative JSON summary.

Clustering also writes `cluster_achievements.json`: each cluster reports its
transition count and all 22 ground-truth achievement counts/percentages. The
denominator is the number of transitions in that cluster; simultaneous unlocks
can make percentages sum above 100%. These labels are diagnostics only and
are not used for encoder training or clustering. New discovery datasets retain
`new_achievements` (a boolean vector per transition). Older datasets still load,
but cannot provide this report without collecting the missing metadata again.

## Windows GPU note

Native Windows JAX uses the CPU backend.  The same source uses CUDA under Linux
or WSL2 when a CUDA-enabled JAX wheel is installed.

## Linux conda execution

The `craftax` conda environment can be reproduced from this directory:

```bash
conda env create -f environment-sea.yml
conda activate craftax
python -m pip check
```

`requirements-sea-cuda.txt` pins JAX 0.4.35 and CUDA 12.1 / cuDNN 9.1
wheels for the NVIDIA 535 driver.  Avoid independently upgrading the CUDA
wheels: newer namespace packages are incompatible with this JAX version's
CUDA path discovery.  Unset `LD_LIBRARY_PATH` if it points to a different CUDA
installation.

Run the primary pixel experiment with independent seeds on GPUs 0 and 1:

```bash
python scripts/run_sea_experiments.py --output runs/sea-pixels --gpus 0 1 --seeds 0 1
```

The supervisor records its PID and child PIDs in `status.json`, freezes a copy
of the source under `source/`, and writes a separate `seed-N/train.log` for
each seed.  It terminates the other child if one run fails.  Sending SIGTERM
to the supervisor stops both runs.  Use a persistent terminal or launch the
supervisor with redirected output and a detached session for unattended runs.

Each PPO stage logs update-local metrics to `progress.jsonl` and atomically
updates `progress.json` and `base_policy_latest.msgpack` or
`goal_policy_latest.msgpack`.  These intermediate files contain policy
parameters, not optimizer or environment state.  The original final policy
and cumulative metric files are still written at stage completion.  The
`--log-interval` training option controls reporting frequency (the two-GPU
supervisor uses 10 updates).  `results.json` collects both seeds' final
training summaries only after both pipelines complete.

## Reuse the collection policy

The episode-buffer fix is retained in the online learner and now evolves while
new rollouts arrive, rather than being built in a separate replay pass. Existing
experiment directories and their frozen source snapshots remain unchanged.

```bash
python scripts/run_sea_experiments.py --output runs/streaming-sea \
  --base-run-root runs/sea-20260911/full --gpus 0 1 --seeds 0 1
```

This trains a fresh encoder and exploration policy using the saved collection
PPO. It retains separate configs, source snapshots, checkpoints and metrics.
