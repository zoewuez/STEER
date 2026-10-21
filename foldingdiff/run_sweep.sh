#!/usr/bin/env bash
# Sample backbones in length chunks and score designability (scTM) for one lam_start.
#
#   bash run_sweep.sh OUTDIR [SAMPLER] [LAM_START]
#
# Each chunk uses a ladder of N replicas with lam_end = lam_start + LAM_STEP * N.
# Environment variables override the defaults below.
set -euo pipefail

OUTDIR="$(realpath -m "${1:?usage: run_sweep.sh OUTDIR [SAMPLER] [LAM_START]}")"
SAMPLER="${2:-steer}"
LAM_START="${3:-0.5}"

FOLDINGDIFF_DIR="${FOLDINGDIFF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/external/foldingdiff}"
LAM_STEP="${LAM_STEP:-0.01}"
NUM_PER_LENGTH="${NUM_PER_LENGTH:-10}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"

FOLDINGDIFF_ENV="${FOLDINGDIFF_ENV:-foldingdiff}"
PROTEINMPNN_ENV="${PROTEINMPNN_ENV:-proteinmpnn}"
OMEGAFOLD_ENV="${OMEGAFOLD_ENV:-omegafold}"
PROTFID_ENV="${PROTFID_ENV:-}"
PROTFID_REFERENCE="${PROTFID_REFERENCE:-cath_test}"

# lo hi n_replicas (hi exclusive)
CHUNKS=(
    "50 60 4"
    "60 70 5"
    "70 80 6"
    "80 90 7"
    "90 100 8"
    "100 110 9"
    "110 128 10"
)

eval "$(conda shell.bash hook)"
BIN="${FOLDINGDIFF_DIR}/bin"
mkdir -p "$OUTDIR"

for spec in "${CHUNKS[@]}"; do
    read -r lo hi n_rep <<< "$spec"
    lam_end=$(awk -v s="$LAM_START" -v d="$LAM_STEP" -v n="$n_rep" 'BEGIN { printf "%.4f", s + d * n }')
    chunk="${OUTDIR}/len_${lo}_${hi}"
    echo "== lengths ${lo}-$((hi - 1)): ${SAMPLER}, ${n_rep} replicas, lam ${LAM_START} -> ${lam_end}"

    sampler_args=(--sampler "$SAMPLER" --lam_start "$LAM_START" --lam_end "$lam_end")
    [ "$SAMPLER" == "steer" ] && sampler_args+=(--n_replicas "$n_rep")

    if [ ! -f "${chunk}/sampled_pdb/generated_0.pdb" ]; then
        rm -rf "$chunk"
        conda activate "$FOLDINGDIFF_ENV"
        python "${BIN}/sample.py" --outdir "$chunk" -n "$NUM_PER_LENGTH" -l "$lo" "$hi" \
            --seed "$SEED" --device "$DEVICE" "${sampler_args[@]}"
        conda deactivate
    fi

    cd "$chunk"
    if [ ! -d proteinmpnn_residues ]; then
        conda activate "$PROTEINMPNN_ENV"
        python "${BIN}/pdb_to_residue_proteinmpnn.py" sampled_pdb
        conda deactivate
    fi
    if [ ! -d omegafold_predictions ]; then
        conda activate "$OMEGAFOLD_ENV"
        python "${BIN}/omegafold_across_gpus.py" proteinmpnn_residues/*.fasta -g 0
        conda deactivate
    fi
    if [ ! -f sctm_scores.json ]; then
        conda activate "$FOLDINGDIFF_ENV"
        python "${BIN}/sctm.py" -f omegafold_predictions
        conda deactivate
    fi
    cd - > /dev/null
done

if [ -n "$PROTFID_ENV" ]; then
    pooled="${OUTDIR}/all_sampled_pdb"
    rm -rf "$pooled" && mkdir -p "$pooled"
    for chunk in "${OUTDIR}"/len_*; do
        for f in "${chunk}"/sampled_pdb/*.pdb; do
            cp "$f" "${pooled}/$(basename "$chunk")_$(basename "$f")"
        done
    done
    conda activate "$PROTFID_ENV"
    echo "yes" | protfid --pdb_dir "$pooled" --output_file "${OUTDIR}/fid.txt" \
        --reference "$PROTFID_REFERENCE"
    conda deactivate
fi

conda activate "$FOLDINGDIFF_ENV"
python "${BIN}/merge_sctm.py" "$OUTDIR" -o "${OUTDIR}/combined_sctm.csv"
