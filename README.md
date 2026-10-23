# EnvTok-MACE

A from-scratch PyTorch implementation of the MACE equivariant interatomic potential,
trained with multi-node DDP on the EnvTok subset of OMat24 (795,721 structures,
about 6.6M atoms), following Moerman et al., *Atoms that Matter* (arXiv:2610.14392).

**Status:** work in progress. The model and the training stack are implemented and
tested; training on NSCC ASPIRE 2A is being set up. See [docs/PLAN.md](docs/PLAN.md).

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
uv sync                       # Python 3.12, torch 2.5.1, e3nn 0.4.4, mace-torch (tests only)
uv run pytest                 # unit, parity and distributed tests
uv run envtok-mace train --config configs/train/debug_cpu.yaml   # tiny end-to-end run
```

Cluster scripts and the data pipeline are described in [docs/NSCC.md](docs/NSCC.md).

## Data

Training data are not redistributed. The subset is rebuilt from
[OMat24](https://huggingface.co/datasets/facebook/OMAT24) (CC BY 4.0) using the
structure IDs published by the authors on
[Zenodo](https://doi.org/10.5281/zenodo.23137852). `tests/fixtures/` holds 40
OMat24 validation structures for the tests.

## References

- E. Moerman, I. Poltavsky, A. Charkin-Gorbulin, A. Tkatchenko. *Atoms that Matter:
  Largest Materials Datasets Are 99% Redundant, and Local Chemistry Can Guide the
  Required 1%.* arXiv:2610.14392 (2026). Code: https://github.com/em819/omat24-chemenv-subset
- I. Batatia et al. *MACE: Higher Order Equivariant Message Passing Neural Networks
  for Fast and Accurate Force Fields.* NeurIPS 2022. Code: https://github.com/ACEsuit/mace
- L. Barroso-Luque et al. *Open Materials 2024 (OMat24) Inorganic Materials Dataset
  and Models.* arXiv:2410.12771 (2024).

## License

MIT (code). Datasets keep their own licenses.
