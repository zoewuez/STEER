#!/usr/bin/env bash
# Train a low-dim diffusion policy on a random subset of the proficient-human demos.
# Usage: scripts/train.sh TASK SUBSAMPLE_RATIO DATASET_HDF5 OUTPUT_DIR [ROBOMIMIC_DIR]
set -euo pipefail

TASK="$1"
RATIO="$2"
DATASET="$3"
OUTPUT_DIR="$4"
ROBOMIMIC_DIR="${5:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/external/robomimic}"

CONFIG="$(mktemp --suffix=.json)"
python - "$ROBOMIMIC_DIR/robomimic/exps/templates/diffusion_policy.json" "$CONFIG" "$TASK" "$OUTPUT_DIR" <<'PY'
import json, sys
template, out, task, output_dir = sys.argv[1:]
cfg = json.load(open(template))
cfg["experiment"]["name"] = f"{task}_diffusion"
cfg["experiment"]["validate"] = True
cfg["experiment"]["rollout"]["enabled"] = False
cfg["train"]["output_dir"] = output_dir
cfg["train"]["hdf5_filter_key"] = "train"
cfg["train"]["hdf5_validation_filter_key"] = "valid"
json.dump(cfg, open(out, "w"), indent=4)
PY

python "$ROBOMIMIC_DIR/robomimic/scripts/subsample_train.py" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --subsample_ratio "$RATIO"
