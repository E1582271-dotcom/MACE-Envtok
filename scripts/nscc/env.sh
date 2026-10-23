# Sourced by every NSCC job script. Reads .env (git-ignored) for account-specific
# settings; nothing account-specific is hard-coded in the repository.
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
[ -f "$REPO_DIR/.env" ] && set -a && . "$REPO_DIR/.env" && set +a
: "${ENVTOK_DATA:?set ENVTOK_DATA in $REPO_DIR/.env}"
: "${ENVTOK_SCRATCH:?set ENVTOK_SCRATCH in $REPO_DIR/.env}"
VENV=${VENV:-$HOME/.venvs/envtok-mace}
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
# Compute nodes are offline: never try to reach the network from a job.
export WANDB_MODE=offline HF_HUB_OFFLINE=1
