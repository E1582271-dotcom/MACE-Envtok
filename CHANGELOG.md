# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Diatomic curves** (`envtok-mace diatomics`, `results/diatomics/`, figure
  `diatomics`). MLIP Arena's homonuclear task covers all 68 elements, for this
  work, the authors' EnvTok model and MACE-OMAT-0. Curves are referenced to two
  isolated atoms. The arena's smoothness metrics are reported per element, with
  counts of single-well curves and of wells deeper than any homonuclear bond.
- **Physics tests:** extensivity under supercells, forces summing to zero,
  locality beyond the cutoff, and smoothness as a pair crosses the cutoff.

### Changed

- The README and the test skip message say how to enable the ten tests that are
  skipped by default: the authors' models from Zenodo and the `bench` extra.
- The README states the single-GPU training throughput against mace-torch, and
  `docs/BENCHMARKS.md` names the file behind the MD speed-ups.
- `results/eval/val10_ours_epoch510_vs_authors.json` is now
  `results/eval/val10_authors_envtok.json`. It keeps only the authors' model,
  which is the reference line in the training figure. The entry for a
  mid-training checkpoint of ours is dropped; the training curve has every epoch.
- The ID/OOD table and the figures read the benchmark summaries through one
  loader (`analysis/results_io.py`) that maps the models' names in the different
  files to one set. The table and every figure are byte-identical to before.
- MatCalc, nebDFT2k and MD stability share their rank placement, wall-clock
  timeout, failure handling and worker spawning (`bench/_runner.py`). Sharding
  and scoring are unchanged.

### Fixed

- `results/matpes/tables/matpes_tables.json` labelled its values meV although
  they are in eV; the labels now say eV. The values and the meV CSV tables are
  unchanged.
- `docs/NSCC.md` gave CUDA builds that `setup_env.sh` no longer offers.

### Removed

- Empty-directory placeholders (`.gitkeep`) from folders that now hold files.

## [1.0.0] - 2026-10-24

Public release with the trained weights.

### Added

- **Weights.** Trained weights attached to the release as `mace-envtok-1.0.0.pt`
  (CC BY 4.0), in the e3nn format that needs only the base dependencies.
  `load_model(path, backend="cueq")` converts them for the GPU kernels.
- **`envtok-mace export-weights`**, which writes a training checkpoint as such a
  portable file, with its provenance.

### Fixed

- A checkpoint written on a GPU node now loads on a machine without
  cuEquivariance's GPU kernels.
  - The kernels change which derived constants cuEquivariance registers, and
    loading was strict about them.
  - These constants are no longer read from checkpoints at all. A saved copy
    carries the float32 precision it was built in, which put errors of up to
    2e-7 relative into float64 runs. The float64 MD-stability and NEB runs made
    on the cluster with the cuEquivariance backend carry these errors, which are
    below the precision of every reported number.
- Every font in the SVG figures now falls back to a sans-serif face, so viewers
  without Arial do not see a serif fallback.

### Removed

- The working plan (`docs/PLAN.md`) from the repository; the README and `docs/`
  describe the project as built.

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

[Unreleased]: https://github.com/E1582271-dotcom/MACE-Envtok/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/E1582271-dotcom/MACE-Envtok/releases/tag/v1.0.0
[0.9.0]: https://github.com/E1582271-dotcom/MACE-Envtok/releases/tag/v0.9.0
