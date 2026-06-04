#!/usr/bin/env bash
set -e

VOXTELL_REPO="${1:-}"

if [ -z "$VOXTELL_REPO" ]; then
  echo "Usage: bash scripts/install_into_voxtell.sh /path/to/VoxTell"
  exit 1
fi

mkdir -p "$VOXTELL_REPO/voxtell/applications"

cp voxtell_l3_bodycomp/l3_body_composition.py \
   "$VOXTELL_REPO/voxtell/applications/l3_body_composition.py"

cp voxtell_l3_bodycomp/__init__.py \
   "$VOXTELL_REPO/voxtell/applications/__init__.py"

echo "Installed VoxTell-L3 module into: $VOXTELL_REPO/voxtell/applications"
