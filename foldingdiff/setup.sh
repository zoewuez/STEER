#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=https://github.com/microsoft/foldingdiff.git
COMMIT=18c865338db937fbd2ffba1e4fdfbcfdb99de8b7
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-${HERE}/external/foldingdiff}"

if [ ! -d "$DEST/.git" ]; then
    git clone "$UPSTREAM" "$DEST"
fi
git -C "$DEST" checkout -q "$COMMIT"
cp -r "$HERE/overlay/." "$DEST/"

cat <<MSG
FoldingDiff with STEER is ready at $DEST

Next steps:
  conda env create -f $DEST/environment.yml && conda activate foldingdiff
  pip install -e $DEST
  pip install -e $(dirname "$HERE")
  cd $DEST && python bin/sample.py --help
MSG
