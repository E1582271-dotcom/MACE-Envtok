# Sourced by job scripts: derive the rank layout from $PBS_NODEFILE (or from a file
# given as $1, e.g. its first k lines) and define run_ranks.
#   run_ranks <command ...>          starts the command once per GPU, as torchrun would
#   RUN_TIMEOUT=10m run_ranks ...    gives up on the launch after 10 minutes
#   RANK_LOG=<prefix> run_ranks ...  each node writes its output to <prefix>.<host>.log
#
# Processes are started with pbsdsh (PBS task manager), one launcher per node, so
# they run inside the job's cgroup on that node and see only the job's GPUs there.
# (Processes started by Cray PALS mpiexec run outside it and see every GPU.)
ranks_layout() {
    local nodefile=${1:-$PBS_NODEFILE}
    NRANKS=$(grep -c . "$nodefile")
    RANK_HOSTS=$(cut -d. -f1 "$nodefile" | awk '!seen[$0]++' | paste -sd, -)
    RANK_NNODES=$(echo "$RANK_HOSTS" | tr ',' '\n' | grep -c .)
    RANK_NODEFILE=$nodefile
    export MASTER_ADDR=${RANK_HOSTS%%,*}
    export NRANKS RANK_HOSTS RANK_NNODES RANK_NODEFILE
}

# Variables handed to every rank (pbsdsh does not forward the caller's environment).
# CUDA_VISIBLE_DEVICES is deliberately not forwarded: each node has its own.
RANK_ENV_VARS="HOME PATH PYTHONUNBUFFERED WANDB_MODE HF_HUB_OFFLINE OMP_NUM_THREADS
MASTER_ADDR MASTER_PORT ENVTOK_DATA ENVTOK_SCRATCH ENVTOK_WALLTIME ENVTOK_PG_TIMEOUT_MIN
NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_GID_INDEX NCCL_IB_HCA NCCL_DEBUG NCCL_DEBUG_SUBSYS
TORCH_NCCL_ASYNC_ERROR_HANDLING RANK_LOG"

run_ranks() {
    local envs=() v idx=0 host seen="," pids="" status=0 p
    for v in $RANK_ENV_VARS; do
        [ -n "${!v+x}" ] && envs+=("$v=${!v}")
    done
    envs+=("PBS_NODEFILE=$RANK_NODEFILE")
    # pbsdsh -n takes an index into the job's node list; RANK_NODEFILE is always a
    # prefix of $PBS_NODEFILE, so its line numbers are valid indices.
    while read -r line; do
        host=${line%%.*}
        case "$seen" in
            *",$host,"*) ;;
            *) seen="$seen$host,"
               timeout --signal=TERM "${RUN_TIMEOUT:-0}" \
                   pbsdsh -n "$idx" -- /usr/bin/env "${envs[@]}" \
                   bash "$REPO_DIR/scripts/nscc/launch_ranks.sh" "$@" &
               pids="$pids $!" ;;
        esac
        idx=$((idx + 1))
    done < "$RANK_NODEFILE"
    for p in $pids; do
        wait "$p" || status=$?
    done
    return $status
}
