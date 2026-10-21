#!/usr/bin/env bash
# Evaluate every sampler on a trained checkpoint with the paper settings for TASK.
# Usage: scripts/eval.sh TASK CHECKPOINT [OUTPUT_DIR] [ROBOMIMIC_DIR]
set -euo pipefail

TASK="$1"
AGENT="$2"
OUTPUT_DIR="${3:-results/$TASK}"
ROBOMIMIC_DIR="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/external/robomimic}"

SAMPLERS=(vanilla tsr cns mcmc steer crepe)
SEEDS=(1 50 100 150 200)
N_ROLLOUTS=50
HORIZON=700

case "$TASK" in
    lift|can|tool_hang) LAM_START=0.9; LAM_END=1.1;  STEER_REPLICAS=4 ;;
    square)             LAM_START=0.9; LAM_END=1.02; STEER_REPLICAS=6 ;;
    *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT_DIR"
for SAMPLER in "${SAMPLERS[@]}"; do
    lam_start=$LAM_START
    n_replicas=1
    [ "$SAMPLER" == "vanilla" ] && lam_start=1.0
    [ "$SAMPLER" == "steer" ] && n_replicas=$STEER_REPLICAS
    for SEED in "${SEEDS[@]}"; do
        python "$ROBOMIMIC_DIR/robomimic/scripts/run_trained_agent.py" \
            --agent "$AGENT" \
            --n_rollouts "$N_ROLLOUTS" \
            --horizon "$HORIZON" \
            --seed "$SEED" \
            --sampler "$SAMPLER" \
            --lam_start "$lam_start" \
            --lam_end "$LAM_END" \
            --n_replicas "$n_replicas" \
            2>&1 | tee "$OUTPUT_DIR/${SAMPLER}_seed${SEED}.log"
    done
done
