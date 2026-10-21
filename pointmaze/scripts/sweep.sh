#!/usr/bin/env bash
# Run from the pointmaze directory produced by setup.sh.
set -uo pipefail

DATASET=${DATASET:-pointmaze-giant-navigate-v0}
DEVICE=${DEVICE:-cuda}
SEEDS=(${SEEDS:-1 50 100 150 200})
SAMPLERS=(${SAMPLERS:-vanilla tsr cns mcmc steer crepe})
N_REPLICAS=(${N_REPLICAS:-2 4 8 16})
LAM_START=${LAM_START:-0.9}
LAM_STEP=${LAM_STEP:-0.01}

export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p sweep_logs

failed=()
run () {
    local tag="$1"; shift
    echo "== $tag"
    if ! python run.py "$@" > "sweep_logs/$tag.log" 2>&1; then
        echo "   failed, see sweep_logs/$tag.log"
        failed+=("$tag")
    fi
}

for sampler in "${SAMPLERS[@]}"; do
    for n in "${N_REPLICAS[@]}"; do
        lam_end=$(awk -v s="$LAM_START" -v d="$LAM_STEP" -v n="$n" 'BEGIN { printf "%.4g", s + d * n }')
        for seed in "${SEEDS[@]}"; do
            run "bon_${sampler}_n${n}_seed${seed}" \
                --dataset "$DATASET" --method bon --device "$DEVICE" --seed "$seed" \
                --sampler "$sampler" --n_replicas "$n" --lam_start "$LAM_START" --lam_end "$lam_end"
        done
    done
done

for method in bfs-resampling dfs; do
    for seed in "${SEEDS[@]}"; do
        run "${method}_vanilla_seed${seed}" \
            --dataset "$DATASET" --method "$method" --device "$DEVICE" --seed "$seed" --sampler vanilla
    done
done

if [ "${#failed[@]}" -gt 0 ]; then
    printf 'failed: %s\n' "${failed[@]}"
    exit 1
fi
