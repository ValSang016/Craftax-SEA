# SEA on Craftax-Classic with PPO-RNN

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
  --discovery-train-steps 10 `
  --discovery-batch-size 32 `
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

The discovery rollout and the saved replay size are separate.  By default the
collector runs for every requested interaction, but transfers at most about
200,000 uniformly sampled transitions plus 100,000 sampled achievement
transitions from the accelerator.  This prevents a 50M-step pixel rollout from
creating a terabyte-scale replay file.  Use
`--discovery-dataset-limit` and `--discovery-positive-limit` to tune this
tradeoff.  Setting `--discovery-dataset-limit` greater than or equal to
`--discovery-interactions` stores every transition and should only be used for
small diagnostics.

## Windows GPU note

Native Windows JAX uses the CPU backend.  The same source uses CUDA under Linux
or WSL2 when a CUDA-enabled JAX wheel is installed.
