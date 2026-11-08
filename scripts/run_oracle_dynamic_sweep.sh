#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 SEED" >&2
  exit 2
fi

seed="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/opt/conda/envs/craftax/bin/python}"
oracle_graph="${ORACLE_GRAPH:-$repo_root/artifacts/oracle_graphs/craftax_classic_22.npz}"
run_root="${RUN_ROOT:-$repo_root/runs/oracle-residual}"
lock_root="${GPU_LOCK_ROOT:-/tmp/craftax-oracle-residual-gpu-locks}"
seed_root="$run_root/seed-$seed"
# Space-separated overrides allow a sweep to stay on a dedicated physical GPU.
read -r -a betas <<< "${BETAS:-1.0 0.8 0.5 0.2}"
read -r -a gpus <<< "${GPUS:-0 1 2 3}"

if [[ ! -f "$oracle_graph" ]]; then
  echo "Missing oracle graph: $oracle_graph" >&2
  exit 1
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python is not executable: $python_bin" >&2
  exit 1
fi

mkdir -p "$seed_root" "$lock_root"
exec > >(tee -a "$seed_root/sweep.log") 2>&1

echo "[$(date -Is)] seed=$seed waiting for one of GPUs: ${gpus[*]}"
selected_gpu=""
lock_fd=""
while [[ -z "$selected_gpu" ]]; do
  for gpu in "${gpus[@]}"; do
    lock_file="$lock_root/gpu-$gpu.lock"
    exec {candidate_fd}>"$lock_file"
    if flock -n "$candidate_fd"; then
      selected_gpu="$gpu"
      lock_fd="$candidate_fd"
      break
    fi
    exec {candidate_fd}>&-
  done
  if [[ -z "$selected_gpu" ]]; then
    sleep 30
  fi
done

echo "[$(date -Is)] seed=$seed acquired physical GPU $selected_gpu"
echo "$$ seed=$seed started=$(date -Is)" > "$lock_root/gpu-$selected_gpu.owner"

cleanup() {
  rm -f "$lock_root/gpu-$selected_gpu.owner"
  eval "exec ${lock_fd}>&-"
}
trap cleanup EXIT

for beta in "${betas[@]}"; do
  beta_tag="${beta/./p}"
  output="$seed_root/beta-$beta_tag"
  wandb_name="oracle-residual-s$seed-beta-$beta"
  wandb_group="oracle-residual-s$seed"

  if [[ -f "$output/goal_metrics.json" ]]; then
    echo "[$(date -Is)] completed output exists, skipping $output"
    continue
  fi
  if [[ -f "$output/config.json" ]]; then
    echo "[$(date -Is)] refusing to overwrite incomplete run: $output" >&2
    exit 1
  fi
  mkdir -p "$output"

  command=(
    "$python_bin" -u -m craftax.sea.train
    --symbolic
    --high-level-controller residual
    --seed "$seed"
    --graph-path "$oracle_graph"
    --no-new-tasks
    --immortal false
    --health-reward-coef 0.1
    --contrast-weight 0.6
    --log-interval 10
    --idle-timeout 10000
    --high-level-goal-timeout 300
    --dynamic-beta "$beta"
    --wandb-entity zaezae01
    --wandb-project craftax-dynamic-graph
    --wandb-mode "${WANDB_MODE:-online}"
    --wandb-name "$wandb_name"
    --wandb-group "$wandb_group"
    --output "$output"
  )

  printf '[%s] starting:' "$(date -Is)"
  printf ' %q' "${command[@]}"
  printf '\n'
  (
    cd "$repo_root"
    env -u LD_LIBRARY_PATH -u PYTHONPATH \
      CUDA_VISIBLE_DEVICES="$selected_gpu" \
      JAX_PLATFORMS=cuda,cpu \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
      OMP_NUM_THREADS=8 \
      OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      PYTHONUNBUFFERED=1 \
      "${command[@]}"
  ) 2>&1 | tee "$output/train.log"
  echo "[$(date -Is)] completed seed=$seed beta=$beta"
done

