#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 RUN_ROOT [native|immortal|mortal] [OUTPUT_DIR]" >&2
  exit 2
fi

RUN_ROOT=${1%/}
ENV_MODE=${2:-native}
OUTPUT_DIR=${3:-videos/jsonl/$(basename "$RUN_ROOT")-${ENV_MODE}}

EPISODES=${EPISODES:-200}
MAX_STEPS=${MAX_STEPS:-10000}
EVAL_SEED=${EVAL_SEED:-100}
PYTHON_BIN=${PYTHON_BIN:-python}

case "$ENV_MODE" in
  native)
    ENV_ARGS=()
    ;;
  immortal)
    ENV_ARGS=(--immortal true)
    ;;
  mortal)
    ENV_ARGS=(--immortal false)
    ;;
  *)
    echo "ENV_MODE must be one of: native, immortal, mortal" >&2
    exit 2
    ;;
esac

if [[ -n ${HEALTH_REWARD_COEF:-} ]]; then
  ENV_ARGS+=(--health-reward-coef "$HEALTH_REWARD_COEF")
fi

required=(
  config.json
  goal_policy.msgpack
  transition_encoder.msgpack
  clusters.npz
)

if [[ -n ${TRAIN_SEEDS:-} ]]; then
  read -r -a train_seeds <<< "$TRAIN_SEEDS"
else
  train_seeds=()
  while IFS= read -r seed_dir; do
    train_seeds+=("${seed_dir##*/seed-}")
  done < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' -printf '%p\n' | sort -V)
fi

if (( ${#train_seeds[@]} == 0 )); then
  echo "No seed directories found under $RUN_ROOT" >&2
  exit 1
fi

missing=()
for train_seed in "${train_seeds[@]}"; do
  run_dir="$RUN_ROOT/seed-$train_seed"
  for filename in "${required[@]}"; do
    if [[ ! -f "$run_dir/$filename" ]]; then
      missing+=("$run_dir/$filename")
    fi
  done
done

if (( ${#missing[@]} > 0 )); then
  echo "Cannot evaluate the selected training seeds. Missing artifacts:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Run root       : $RUN_ROOT"
echo "Environment    : $ENV_MODE"
echo "Training seeds : ${train_seeds[*]}"
echo "Episodes/seed  : $EPISODES"
echo "Evaluation seed: $EVAL_SEED (shared across all checkpoints)"
echo "Output         : $OUTPUT_DIR"

for train_seed in "${train_seeds[@]}"; do
  run_dir="$RUN_ROOT/seed-$train_seed"
  output="$OUTPUT_DIR/seed-$train_seed.jsonl"

  if [[ -e "$output" && ${OVERWRITE:-0} != 1 ]]; then
    echo "Refusing to overwrite $output (set OVERWRITE=1 to replace it)." >&2
    exit 1
  fi

  echo
  echo "[training seed $train_seed] writing $output"
  "$PYTHON_BIN" -u scripts/replay_sea_policy.py "$run_dir" \
    --checkpoint goal \
    --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" \
    --seed "$EVAL_SEED" \
    --no-window \
    --log "$output" \
    "${ENV_ARGS[@]}"
done

echo
echo "Finished: $OUTPUT_DIR"
