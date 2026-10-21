#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_URL=https://github.com/XiangchengZhang/Diffusion-inference-scaling.git
UPSTREAM_COMMIT=c158ab9eb058adb24ad0f8d48506ab837d7fae91

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/external/Diffusion-inference-scaling}"

if [ ! -d "$DEST/.git" ]; then
    git clone "$UPSTREAM_URL" "$DEST"
fi
git -C "$DEST" checkout -q "$UPSTREAM_COMMIT"
cp -r "$HERE/overlay/." "$DEST/pointmaze/"
cp "$HERE/scripts/sweep.sh" "$DEST/pointmaze/"

cat <<MSG
PointMaze code ready in $DEST/pointmaze

Next steps:
  cd $DEST/pointmaze
  pip install -r requirements.txt
  pip install -e ogbench
  pip install -e .
  pip install -e $(cd "$HERE/.." && pwd)
  # put the pretrained checkpoints in logs/ (see README)
  python run.py --dataset pointmaze-giant-navigate-v0 --method bon --sampler steer \\
      --lam_start 0.9 --lam_end 0.94 --n_replicas 4 --seed 1
MSG
