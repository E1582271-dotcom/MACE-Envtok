# MACE-Envtok

A from-scratch PyTorch implementation of the MACE equivariant interatomic potential,
trained with multi-node DDP on the EnvTok subset of OMat24 (795,721 structures,
about 6.6M atoms), following Moerman et al., *Atoms that Matter* (arXiv:2610.14392).

**Status:** the model is trained and evaluated on every benchmark of the paper:
OMat24 validation, MatPES, MatCalc-Bench, MD stability and Li-ion migration
barriers ([docs/MATPES.md](docs/MATPES.md), [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

## Results

Trained from random initialisation on 8 A100 GPUs, each on a different node of
NSCC ASPIRE 2A. The run stopped early at epoch 696 after 22 h of wall-clock time,
about 176 GPU-hours. The comparison models are the authors' released ones.

**OMat24 validation, in distribution.** These 591,981 frames were used neither
for training nor for model selection. MAE in meV/atom, meV/Å and meV/Å³.

| Model | Training data | Energy | Force | Stress |
|---|---|---|---|---|
| **This repository, EnvTok subset** | 795,721 structures | **24.59** | **109.4** | **2.879** |
| Authors, EnvTok subset | same | 25.47 | 112.1 | 2.946 |
| Authors, random subsets (4 seeds) | same size | 26.69 ± 0.17 | 116.8 ± 0.7 | 3.08 |
| Authors, DIRECT subset | same size | 28.67 | 118.6 | 3.125 |
| MACE-OMAT-0 | all of OMat24 | 17.47 | 84.5 | 2.577 |

The authors selected their models on the full validation set, which contains
these frames, so this comparison favours them.

**MatPES PBE 2025.2, out of distribution.** These are 323,096 structures from a
different dataset, binned as in the paper. Per-structure force MAE in meV/Å,
given as median / pooled mean.

| Model | Bulk (242,319) | Non-bulk (80,766) |
|---|---|---|
| **This repository** | **110.1** / **143.3** | 240.4 / 320.8 |
| Authors, EnvTok | 112.6 / 148.6 | 251.7 / 3,904 |
| Authors, random (4 seeds) | 114.2–115.9 / 149.7–151.7 | 246.2–254.6 / 1,126–137,634 |
| Authors, DIRECT | 119.3 / 151.2 | 228.7 / 327.1 |
| MACE-OMAT-0 | 83.6 / 126.4 | 158.9 / 275.4 |

Rescoring the authors' EnvTok model with this pipeline reproduces their
published per-atom errors. The median difference is 1.5e-6 eV/Å, and their
table values are recovered to every printed digit
([docs/MATPES.md](docs/MATPES.md)).

The pooled non-bulk mean is set by a few structures where a model's force error
explodes. Atoms with errors above 10 eV/Å: 4,693 for the authors' EnvTok model,
1,481 here. Across the four random seeds, the same mean varies more than
100-fold, so the medians are the comparable numbers.

Each subset has one training run here and one from the authors. The gap to their
EnvTok model is larger than the spread of their random-subset seeds, but one run
on each side is not enough to explain it.

**MatCalc-Bench, materials properties.** Median errors against DFT; formation
energy as the paper's MAE.

| Model | E_form MAE (meV/atom) | Bulk modulus | Shear modulus | C_V (300 K) | Force softening |
|---|---|---|---|---|---|
| **This repository** | 62.9 | **+1.4 %** | −4.9 % | +0.41 % | 3.6 % |
| Authors, EnvTok | 62.7 | +1.5 % | −5.8 % | +0.52 % | 3.8 % |
| Authors, random (4 seeds) | 65.5 | +2.7 % | −5.6 % | +0.38 % | 3.3 % |
| Authors, DIRECT | 74.3 | +2.0 % | −8.0 % | +0.44 % | 3.9 % |
| MACE-OMAT-0 | 56.0 | +3.2 % | +1.4 % | +0.13 % | 2.4 % |

**MD stability.** 80 NPT runs from 300 K and 0 GPa to 3,000 K and 500 GPa over
10 ps. Completion is scored at the authors' step speed, because the protocol's
wall-clock limit truncates slow runs.

| Model | Mean completion | Physically diverged |
|---|---|---|
| **This repository** | **96.1 %** | **0 / 80** |
| Authors, EnvTok | 92.1 % | 6 / 80 |
| Authors, DIRECT | 92.9 % | 9 / 80 |
| Authors, random (4 seeds) | 81.1 % | 85 / 320 |
| MACE-OMAT-0 | 75.4 % | 31 / 80 |

**Li-ion migration barriers (nebDFT2k).** 1,656 hops, MAE on converged bands. The
reference is PBE without Hubbard U, so hosts that need U are a separate column.

| Model | Without U (eV) | With U (eV) |
|---|---|---|
| **This repository** | 0.138 | **0.259** |
| Authors, EnvTok | 0.124 | 0.269 |
| Authors, random (4 seeds) | 0.148 ± 0.037 | 0.264 |
| Authors, DIRECT | 0.116 | 0.316 |
| MACE-OMAT-0 | 0.146 | 0.270 |

Every benchmark pipeline was checked first by running the authors' released model
through it. Elastic moduli, heat capacities, softening scales and NEB barriers
match their published values to between 1e-13 and 2e-10 relative
([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

## What is implemented here

| Part | Where | Checked by |
|---|---|---|
| Radial basis, Agnesi transform, cutoff, ZBL repulsion | `src/envtok_mace/model/radial.py` | parity with mace-torch |
| Generalised Clebsch-Gordan coupling tensors | `model/cg.py` | equal to mace-torch's (difference 0) |
| Symmetric contraction (product basis), density-normalised interactions | `model/blocks.py` | parity, symmetry tests |
| Model, forces and stress by autograd | `model/mace.py` | finite differences, rotations |
| cuEquivariance backend (fused kernels) | `model/backends.py` | equal to the e3nn backend |
| OMat24 extraction, neighbour lists, batching | `data/` | comparison with ASE and mace-torch |
| Loss, optimiser groups, EMA | `train/loss.py`, `train/optim.py` | equal to mace-torch's loss and groups |
| DDP training, load-balanced batches, gradient accumulation, exact resume | `train/` | multi-process equivalence and resume tests |

- **Parity:** with the same weights, the model reproduces the reference mace-torch
  implementation to 1e-13 in float64, including the authors' released EnvTok model
  ([docs/PARITY.md](docs/PARITY.md)).
- **DDP equivalence:** one process with 4-step gradient accumulation, two processes
  with 2 steps and four processes give the same weights after training.
- **Exact resume:** resuming from a checkpoint is bitwise identical to an
  uninterrupted run.

Spherical harmonics, Wigner 3j symbols and the basic equivariant layers come from
[e3nn](https://github.com/e3nn/e3nn); the fused GPU kernels from
[cuEquivariance](https://github.com/NVIDIA/cuEquivariance). mace-torch is used only
as a reference in the tests.

## Quick start (CPU)

```bash
uv sync                       # Python 3.12, torch 2.7.1, e3nn 0.4.4, mace-torch (tests only)
uv run pytest                 # unit, parity and distributed tests
uv run envtok-mace train --config configs/train/debug_cpu.yaml   # tiny end-to-end run
```

Cluster scripts and the data pipeline are described in [docs/NSCC.md](docs/NSCC.md).

## Data

Training data are not redistributed. The subset is rebuilt from
[OMat24](https://huggingface.co/datasets/facebook/OMAT24) (CC BY 4.0) using the
structure IDs published by the authors on
[Zenodo](https://doi.org/10.5281/zenodo.23137852). `tests/fixtures/` holds 40
OMat24 validation structures for the tests. The MatPES test set is rebuilt from
the [MatPES](https://huggingface.co/datasets/materialyze/matpes) release with
`scripts/data/download_matpes.sh` and `envtok-mace matpes-build`.

## References

- E. Moerman, I. Poltavsky, A. Charkin-Gorbulin, A. Tkatchenko. *Atoms that Matter:
  Largest Materials Datasets Are 99% Redundant, and Local Chemistry Can Guide the
  Required 1%.* arXiv:2610.14392 (2026). Code: https://github.com/em819/omat24-chemenv-subset
- I. Batatia et al. *MACE: Higher Order Equivariant Message Passing Neural Networks
  for Fast and Accurate Force Fields.* NeurIPS 2022. Code: https://github.com/ACEsuit/mace
- L. Barroso-Luque et al. *Open Materials 2024 (OMat24) Inorganic Materials Dataset
  and Models.* arXiv:2410.12771 (2024).
- A. D. Kaplan et al. *A Foundational Potential Energy Surface Dataset for
  Materials (MatPES).* arXiv:2503.04070 (2025).
- MatCalc and MatCalc-Bench: https://github.com/materialsvirtuallab/matcalc
- MLIP Arena stability benchmark (RM24 structures):
  https://huggingface.co/datasets/atomind/mlip-arena
- A. Dembitskiy et al. *Benchmarking machine learning models for predicting lithium
  ion migration (LiTraj).* npj Comput. Mater. 11, 131 (2025),
  doi:10.1038/s41524-025-01571-z

## License

MIT (code). Datasets keep their own licenses.
