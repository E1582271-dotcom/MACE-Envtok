#!/bin/bash
# Download the MatPES PBE 2025.2 release (one JSON record per line, 2.36 GB) and
# verify it against the checksum Hugging Face publishes for the file.
# Run on a machine with internet (on NSCC: the login node).
#   bash scripts/data/download_matpes.sh [DEST_DIR]
set -euo pipefail
DEST=${1:-${ENVTOK_SCRATCH:?set ENVTOK_SCRATCH or pass a directory}/matpes}
FILE=MatPES-PBE-2025.2.jsonl
URL=https://huggingface.co/datasets/materialyze/matpes/resolve/main/$FILE
SHA256=5a225b1bc5c85523da8a2252a606ed4a5a0a8f1eca4615500d57d3185b10660d
SIZE=2358285238

mkdir -p "$DEST"
cd "$DEST"
if [ ! -f "$FILE" ] || [ "$(stat -c %s "$FILE" 2>/dev/null || stat -f %z "$FILE")" != "$SIZE" ]; then
    curl -fsSL --retry 5 -C - -o "$FILE" "$URL"
fi
echo "$SHA256  $FILE" | sha256sum -c - 2>/dev/null || echo "$SHA256  $FILE" | shasum -a 256 -c -
echo "MatPES ready: $DEST/$FILE"
