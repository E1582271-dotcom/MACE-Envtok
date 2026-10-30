# Running on NSCC ASPIRE 2A

Account-specific values (PBS project, paths) live in `.env`, which is git-ignored.
Compute nodes have no internet access, so every download happens on a login node.

## 0. Get the code onto the cluster

Either clone the GitHub repository, or copy it from your laptop:

```bash
rsync -av --exclude local --exclude .venv --exclude .git ~/Desktop/EnvTok-MACE/ <nscc-host>:~/EnvTok-MACE/
```

Then on the login node:

```bash
cd ~/EnvTok-MACE
cp .env.example .env    # fill in NSCC_PROJECT, ENVTOK_DATA (under $HOME), ENVTOK_SCRATCH (under scratch)
```

`ENVTOK_DATA` holds the processed HDF5 files (~15 GB) and must be outside scratch,
which is purged after 30 days without access.

## 1. Python environment (login node)

```bash
TORCH_CUDA=cu126 bash scripts/nscc/setup_env.sh
```

`cu126` is the default and `cu128` the alternative. The cuEquivariance kernels need
CUDA 12.5 or newer, so check the driver version the probe in step 2 reports and pick
the newest build it supports.

## 2. Probe the cluster (15 min, 1 GPU)

```bash
. scripts/nscc/env.sh                      # loads .env into the shell
qstat -Q                                   # queue names and limits
qsub -P "$NSCC_PROJECT" scripts/nscc/probe_gpu.pbs
```

The job log reports the GPU driver version, which MPI launcher `mpiexec` is and
which rank variables it sets, the network interfaces (expect `hsn0`), and
whether an OFI plugin for NCCL is installed. These decide the 8-GPU launcher.

## 3. Download data (login node, resumable)

```bash
. scripts/nscc/env.sh
bash scripts/data/download_zenodo.sh local/zenodo subsets.zip models.zip
nohup bash scripts/data/download_omat24.sh "$ENVTOK_SCRATCH/raw/omat24" both 4 > omat24_download.log 2>&1 &
```

OMat24 is ~198 GB compressed (train 196 GB, val 2 GB). Each archive is unpacked
into its own folder and deleted once unpacked; rerun the command to resume.

## 4. Build the training and validation sets (CPU job)

```bash
qsub -P "$NSCC_PROJECT" scripts/nscc/extract.pbs
```

The job fails loudly unless:

- all 795,721 EnvTok IDs are found;
- the validation set has exactly 657,756 frames after removing rare earths;
- the training set's average neighbour count is 40.5113, as in the authors' log.

The raw OMat24 download can be deleted after this job succeeds.

## Queues (ASPIRE 2A, checked 2026-10-22)

Jobs go to the routing queue `normal`, which picks the execution queue from the
requested GPUs and walltime:

| Queue | GPUs | Walltime | Notes |
|---|---|---|---|
| `gdev` | 1-4 | ≤ 2 h | at most 16 GPUs running in the whole queue, so waits can be long |
| `g1` | 1 | 2:00:01-24 h | one running job per user |
| `g2` | 2-3 | 2:00:01-24 h | one running job per user |
| `g3` | 4 | 2:00:01-24 h | one running job per user |
| `g4` | ≥ 5 | 2:00:01-24 h | the 8-GPU pilot; one running job per user |
| `glong` | 1-64 | 24:00:01-120 h | the main training run; one running job per user |

A short single-GPU job usually starts sooner with `-l walltime=02:00:01` (g1)
than in `gdev`.

Hardware and software seen by the probe job: A100-SXM4-40GB, driver 570.124.06
(CUDA 12.8), Cray PALS `mpiexec` (sets `PMI_RANK`, `PMI_SIZE`,
`PALS_LOCAL_RANKID`), Slingshot interfaces `hsn0`/`hsn1`, no NCCL OFI plugin
(NCCL uses sockets over `hsn*`).
