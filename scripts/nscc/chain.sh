#!/usr/bin/env bash
# Submit a training job followed by N-1 dependent restarts (each starts when the
# previous one ends, for any reason, and resumes from last.pt; a finished run
# exits immediately).
#   bash scripts/nscc/chain.sh <n_jobs> <config> [extra qsub args...]
set -euo pipefail
n=${1:?n_jobs}; config=${2:?config}; shift 2
cd "$(dirname "$0")/../.."
. scripts/nscc/env.sh
prev=$(qsub -P "$NSCC_PROJECT" -v CONFIG="$config" "$@" scripts/nscc/train_8gpu.pbs)
echo "submitted $prev"
for _ in $(seq 2 "$n"); do
    prev=$(qsub -P "$NSCC_PROJECT" -W depend=afterany:"$prev" -v CONFIG="$config" "$@" scripts/nscc/train_8gpu.pbs)
    echo "submitted $prev (after previous)"
done
