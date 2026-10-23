#!/usr/bin/env bash
# Download files from the authors' Zenodo record (10.5281/zenodo.23137852),
# verify them against MD5SUMS.txt and unpack each archive into its own folder.
#
# usage: scripts/data/download_zenodo.sh <dest_dir> [file ...]
#   default files: subsets.zip models.zip
#   others: benchmarks_matpes.zip benchmarks_matcalc.zip benchmarks_md-stability.zip
#           benchmarks_nebdft2k.zip latent-space.zip
set -euo pipefail

RECORD=23137852
BASE="https://zenodo.org/records/${RECORD}/files"

dest=${1:?usage: download_zenodo.sh <dest_dir> [file ...]}
shift || true
files=("$@")
[ ${#files[@]} -eq 0 ] && files=(subsets.zip models.zip)

mkdir -p "$dest"
cd "$dest"

for meta in README.md MD5SUMS.txt; do
    [ -f "$meta" ] || curl -fsSL -o "$meta" "${BASE}/${meta}?download=1"
done

md5_of() {
    if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

for f in "${files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "downloading $f"
        curl -fsSL --retry 5 -C - -o "$f.part" "${BASE}/${f}?download=1"
        mv "$f.part" "$f"
    fi
    out="${f%.zip}"
    if [ ! -d "$out" ]; then
        mkdir "$out"
        unzip -q "$f" -d "$out"
    fi
    # MD5SUMS.txt lists the files inside the archives ("<md5>  ./<path>").
    n_ok=0
    while read -r sum path; do
        rel="${path#./}"
        [ -f "$out/$rel" ] || continue
        if [ "$(md5_of "$out/$rel")" != "$sum" ]; then
            echo "ERROR: md5 mismatch for $out/$rel" >&2
            exit 1
        fi
        n_ok=$((n_ok + 1))
    done < MD5SUMS.txt
    if [ "$n_ok" -eq 0 ]; then
        echo "ERROR: no file of $f matched MD5SUMS.txt" >&2
        exit 1
    fi
    echo "md5 ok: $f ($n_ok files)"
done
