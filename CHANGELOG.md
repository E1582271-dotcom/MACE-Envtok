# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.9.0] - 2026-10-24

First complete state: the model is implemented, trained and evaluated on every
benchmark of the reproduced paper. Kept private until the 1.0.0 release.

### Added

- **Model.** MACE written from scratch in PyTorch with e3nn and cuEquivariance
  backends: Agnesi-transformed Bessel radial basis, density-normalised
  interactions, symmetric contraction over generalised Clebsch–Gordan
  tensors, ZBL repulsion, forces and stress by autograd. Matches mace-torch
  0.3.14 to 1e-13 in float64 with the authors' released weights
  ([docs/PARITY.md](docs/PARITY.md)).
- **Data.** OMat24 subset extraction by structure id, neighbour lists, flat
  HDF5 storage and load-balanced batching.
- **Training.** Multi-node DDP with loss normalisation across ranks and
  micro-batches. The global batch is independent of GPU count, and resume is
  exact, including on a different number of GPUs. Also EMA, AMSGrad,
  ReduceLROnPlateau, early stopping, walltime and NaN guards.
- **Cluster.** NSCC ASPIRE 2A job scripts for single GPUs scattered over
  several nodes ([docs/NSCC.md](docs/NSCC.md)).
- **Results.** Training run on 8 A100 GPUs, 22 h, early stop at epoch 696.
  Evaluation on held-out OMat24 validation data, MatPES, MatCalc-Bench, MD
  stability and nebDFT2k, each validated first against the authors' published
  results ([docs/MATPES.md](docs/MATPES.md),
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
- **Analysis.** An ID / dataset-shift / OOD summary table, and force error
  against latent-space distance from the training data.
- **Figures.** Eight figures drawn from `results/` by `envtok-mace figures`,
  with an automatic layout check.
- **Repository.** CI for tests and lint, pre-commit hooks (ruff),
  `CITATION.cff` and this changelog.

### Fixed

- The element lookup overran its table for Z ≥ 84, a device-side assertion on
  GPU that left the process unable to use CUDA.
- Benchmark workers now stop on a CUDA device error instead of marking all
  their remaining work as failed.
- MatCalc phonon results: pinned phonors 0.2.1 (0.6 makes every phonon
  calculation NaN with phonopy 4.1.0), one worker per GPU against
  out-of-memory failures, and every calculator exception is logged.

[Unreleased]: https://github.com/E1582271-dotcom/MACE-Envtok/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/E1582271-dotcom/MACE-Envtok/releases/tag/v0.9.0
