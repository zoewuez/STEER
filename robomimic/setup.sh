#!/usr/bin/env bash
set -euo pipefail

ROBOMIMIC_COMMIT=ae5799f0fae05c4559ee1f9645b0f77eb5251929
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/external/robomimic}"

if [ ! -d "$DEST/.git" ]; then
    git clone https://github.com/ARISE-Initiative/robomimic.git "$DEST"
fi
git -C "$DEST" checkout -q "$ROBOMIMIC_COMMIT"
cp -r "$HERE/overlay/." "$DEST/"

echo "robomimic with STEER is ready in $DEST"
echo "Next:"
echo "  pip install -e $DEST"
echo "  pip install -e $HERE/.."
