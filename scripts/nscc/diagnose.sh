#!/usr/bin/env bash
# Read-only health check of an NSCC account: prints information, changes nothing.
# Run from your laptop through your own SSH login:
#   ssh <nscc-host> 'bash -l -s' < scripts/nscc/diagnose.sh
section() { printf '\n== %s\n' "$1"; }

section "identity"
whoami; hostname; date

section "projects and balance (myprojects)"
myprojects 2>&1 | head -40

section "storage (myquota)"
myquota 2>&1 | head -20

section "groups"
groups

section "home and scratch"
echo "HOME=$HOME"
for d in "/scratch/users/nus/$USER" "/scratch/$USER"; do [ -d "$d" ] && ls -ld "$d"; done
df -h "$HOME" | tail -1

section "queues (qstat -Q)"
qstat -Q 2>&1 | head -40

section "my jobs"
qstat -u "$USER" 2>&1 | tail -10

section "modules"
if command -v module >/dev/null 2>&1; then
    module avail python miniforge cuda singularity apptainer 2>&1 | head -40
else
    echo "module command not available in this shell"
fi

section "tools"
for t in uv python3 git curl singularity apptainer mpiexec; do
    printf '%-12s %s\n' "$t" "$(command -v "$t" || echo missing)"
done
python3 --version 2>&1

section "outbound internet from this login node (HTTP status)"
for u in https://pypi.org/simple/torch/ \
         https://download.pytorch.org/whl/cu124 \
         https://zenodo.org/records/23137852 \
         https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300-subsampled.tar.gz \
         https://github.com; do
    printf '%-100s %s\n' "$u" "$(curl -s -o /dev/null -I -L -m 20 -w '%{http_code}' "$u")"
done

section "project checkout"
if [ -d "$HOME/EnvTok-MACE" ]; then
    ls "$HOME/EnvTok-MACE"
    [ -f "$HOME/EnvTok-MACE/.env" ] && echo ".env present" || echo ".env missing"
else
    echo "~/EnvTok-MACE not found"
fi
