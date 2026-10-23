#!/bin/bash
# Started once per node (mpiexec --ppn 1). Starts one process per GPU that this
# job owns on this node, numbered consistently across nodes.
#
# $PBS_NODEFILE has one line per rank (chunks of 1 GPU with mpiprocs=1), so a node
# with k of the job's GPUs appears k times; nodes may own different numbers of GPUs.
# Global ranks follow the order of first appearance, so rank 0 is on the first node.
set -uo pipefail
# pbsdsh starts tasks in $HOME; run from the repository so relative paths work
cd "$(dirname "$0")/../.."
me=$(hostname -s)
# Output of pbsdsh tasks does not come back through pbsdsh; write it to a file per node.
if [ -n "${RANK_LOG:-}" ]; then
    exec >> "$RANK_LOG.$me.log" 2>&1
fi
nodes=$(cut -d. -f1 "$PBS_NODEFILE" | awk '!seen[$0]++')
offset=0
mine=0
for n in $nodes; do
    c=$(cut -d. -f1 "$PBS_NODEFILE" | grep -cx "$n")
    if [ "$n" = "$me" ]; then mine=$c; break; fi
    offset=$((offset + c))
done
if [ "$mine" -eq 0 ]; then
    echo "launch_ranks: $me is not in $PBS_NODEFILE" >&2
    exit 1
fi
export WORLD_SIZE=$(grep -c . "$PBS_NODEFILE")
# mpiexec copies the launching node's environment, including its CUDA_VISIBLE_DEVICES
# (GPU UUIDs of *that* node). Rebuild it from the GPUs this job owns here: PBS confines
# the job to its GPUs with cgroups, so nvidia-smi lists exactly those.
# (Runs on fewer GPUs than allocated, e.g. scaling tests, use the first ones.)
if command -v nvidia-smi >/dev/null 2>&1; then
    uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n "$mine" | paste -sd, -)
    n_gpus=$(echo "$uuids" | tr ',' '\n' | grep -c .)
    if [ "$n_gpus" -lt "$mine" ]; then
        echo "launch_ranks: $me shows only $n_gpus GPU(s) but should run $mine rank(s)" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=$uuids
fi
echo "launch_ranks: $me starts ranks $offset..$((offset + mine - 1)) of $WORLD_SIZE" \
     "(MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset}" \
     "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
pids=""
i=0
while [ "$i" -lt "$mine" ]; do
    RANK=$((offset + i)) LOCAL_RANK=$i "$@" &
    pids="$pids $!"
    i=$((i + 1))
done
status=0
for p in $pids; do
    wait "$p" || status=$?
done
exit $status
