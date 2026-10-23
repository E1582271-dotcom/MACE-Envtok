#!/usr/bin/env bash
# Download the OMat24 train split (11 sub-datasets, ~196 GB) and the corrected
# val split (20 Dec 2024, ~2 GB), unpack each archive into its own folder and
# remove the archive once unpacked. Resumable: rerun after an interruption.
#
# usage: scripts/data/download_omat24.sh <dest_dir> [train|val|both] [parallel downloads]
set -euo pipefail
dest=${1:?usage: download_omat24.sh <dest_dir> [train|val|both] [jobs]}
which=${2:-both}
jobs=${3:-4}

BASE=https://dl.fbaipublicfiles.com/opencatalystproject/data/omat
SUBSETS="rattled-1000 rattled-1000-subsampled rattled-500 rattled-500-subsampled rattled-300
rattled-300-subsampled aimd-from-PBE-1000-npt aimd-from-PBE-1000-nvt aimd-from-PBE-3000-npt
aimd-from-PBE-3000-nvt rattled-relax"

fetch() {  # <url> <split dir>
    local url=$1 dir=$2 name tgz want got
    name=$(basename "$url" .tar.gz)
    [ -f "$dir/$name/.done" ] && { echo "have $dir/$name"; return; }
    mkdir -p "$dir"
    tgz="$dir/$name.tar.gz"
    want=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub("\r","",v); print v}')
    curl -sfL --retry 10 --retry-delay 30 -C - -o "$tgz" "$url" || true
    got=$(stat -c %s "$tgz" 2>/dev/null || stat -f %z "$tgz")
    if [ "$want" != "$got" ]; then echo "ERROR: $tgz is $got bytes, expected $want" >&2; return 1; fi
    tar -xzf "$tgz" -C "$dir"
    touch "$dir/$name/.done"
    rm -f "$tgz"
    echo "unpacked $dir/$name"
}
export -f fetch

{
  if [ "$which" != val ]; then for s in $SUBSETS; do echo "$BASE/241018/omat/train/$s.tar.gz $dest/train"; done; fi
  if [ "$which" != train ]; then for s in $SUBSETS; do echo "$BASE/241220/omat/val/$s.tar.gz $dest/val"; done; fi
} | xargs -P "$jobs" -L 1 bash -c 'fetch "$0" "$1"'
echo "OMat24 download complete under $dest"
