#!/usr/bin/env bash
# Build the Python environment on an NSCC login node (compute nodes are offline).
#
# usage: TORCH_CUDA=cu126 bash scripts/nscc/setup_env.sh
#   TORCH_CUDA: cu126 (default) | cu128 -- needs CUDA >= 12.5 for the cuEquivariance kernels -- pick the newest the GPU driver supports;
#   check with `nvidia-smi` in a GPU job (scripts/nscc/probe_gpu.pbs).
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)
VENV=${VENV:-$HOME/.venvs/envtok-mace}
TORCH_CUDA=${TORCH_CUDA:-cu126}
[ -f "$REPO_DIR/.env" ] && set -a && . "$REPO_DIR/.env" && set +a
# keep the package cache (several GB) off the 50 GB home quota
[ -n "${ENVTOK_SCRATCH:-}" ] && export UV_CACHE_DIR=${UV_CACHE_DIR:-$ENVTOK_SCRATCH/.uv-cache}

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Pin versions in every install step; otherwise the resolver may "upgrade" torch
# to a build the cluster's GPU driver cannot run.
CU_MAJOR=${TORCH_CUDA:2:2}
CONSTRAINTS="$VENV.constraints.txt"
mkdir -p "$(dirname "$VENV")"
cat > "$CONSTRAINTS" <<EOF
torch==2.7.1
e3nn==0.4.4
cuequivariance==0.8.1
cuequivariance-torch==0.8.1
cuequivariance-ops-torch-cu${CU_MAJOR}==0.8.1
mace-torch==0.3.14
EOF

uv venv --clear --python 3.12 --python-preference only-managed "$VENV"  # login-node system Pythons do not exist on compute nodes
PY="$VENV/bin/python"
export UV_LINK_MODE=copy
uv pip install --python "$PY" -c "$CONSTRAINTS" "torch==2.7.1" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
uv pip install --python "$PY" -c "$CONSTRAINTS" -e "$REPO_DIR[track]"
uv pip install --python "$PY" -c "$CONSTRAINTS" cuequivariance-torch "cuequivariance-ops-torch-cu${CU_MAJOR}"
# Reference implementation (parity tests, speed baseline) and test tools.
uv pip install --python "$PY" -c "$CONSTRAINTS" mace-torch pytest scikit-learn ase-db-backends

"$PY" - <<'PY'
import torch, e3nn, mace
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| e3nn", e3nn.__version__, "| mace", mace.__version__)
assert torch.__version__.startswith("2.7.1"), "torch was replaced by another version"
try:
    import cuequivariance_torch as cuet
    print("cuequivariance-torch", cuet.__version__)
except Exception as exc:
    print("cuequivariance-torch not available:", exc)
PY
echo "environment ready: $VENV"
