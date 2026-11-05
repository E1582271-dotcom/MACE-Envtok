# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-11-05

### Added

- **Per-structure predictions.** `eval --save-predictions DIR` and
  `matpes --save-predictions` write every structure's predicted and DFT energy,
  forces and stress to an HDF5 file per model (`eval/predictions.py`). The
  reported metrics do not change.
- **`envtok-mace accuracy`** reduces those files to small tables for
  `results/accuracy/`: parity histograms, error quantiles, and errors per
  element, per OMat24 sub-dataset and against the DFT force magnitude.
  `scripts/nscc/accuracy.pbs` runs the whole chain on the cluster.
- **Accuracy against DFT.** Tables and six figures for the held-out OMat24 frames
  and MatPES, for this work, the authors' EnvTok model and MACE-OMAT-0: parity
  plots of energy, forces and stress, force error per element on a periodic table,
  errors per OMat24 sub-dataset next to its share of the training set, and force
  error against the size of the force. The totals reproduce the published tables
  to better than 1e-7 on OMat24 and 2e-5 on MatPES; no earlier number changes.
- `envtok-mace learning-curves` reduces the trainer's step log to a per-epoch
  training loss. It also collects the evaluations of the checkpoints archived
  every 25 epochs on fixed training and held-out subsets, so the training figure
  can show train against validation loss, and train/validation/test errors
  against epoch.
- **Diatomic curves** (`envtok-mace diatomics`, `results/diatomics/`, figure
  `diatomics`). MLIP Arena's homonuclear task covers all 68 elements, for this
  work, the authors' EnvTok model and MACE-OMAT-0. Curves are referenced to two
  isolated atoms. The arena's smoothness metrics are reported per element, with
  counts of single-well curves and of wells deeper than any homonuclear bond.
- **Equation of state** (`envtok-mace eos`, `results/eos/`, figure `eos`). MLIP
  Arena's EOS steps on its WBM set, for this work, the authors' EnvTok model and
  MACE-OMAT-0: the 610 of 1,000 structures with every element in the model's table.
  The arena's smoothness metrics are reported per structure. The Birch-Murnaghan
  bulk modulus is compared with MACE-OMAT-0, because the set has no DFT moduli.
- **NVE drift** (`envtok-mace nve`, `results/nve/`, figure `nve_drift`). The
  total-energy drift over 10 ps on 20 WBM supercells, in float32 and float64, for
  the same three models. Runs start at a nominal 300 and 1000 K and are shown at
  their measured temperature, because some structures heat up well above their
  start in all three models.
- **Phonons and thermal conductivity** (`envtok-mace kappa`,
  `scripts/nscc/kappa_forces.pbs` and `kappa_conductivity.pbs`, `results/kappa/`,
  figure `kappa`). Matbench Discovery's κ_SRME protocol on its 103 phononDB-PBE
  binaries with phono3py 4.1.0, for this work, the authors' EnvTok model,
  MACE-OMAT-0 and MACE-MPA-0. Forces run on a GPU and the Wigner transport
  equation on CPU cores. Each structure also gets ω_max and the heat capacity,
  free energy and entropy at 300 K against the DFT phonons. MACE-MPA-0 reproduces
  its published κ_SRME (0.4131 against 0.4123), κ_SRE (0.2048) and failure rate (0).
- **WBM relaxations** (`envtok-mace wbm`, `scripts/nscc/wbm_8gpu.pbs`,
  `results/wbm/`, figure `wbm`). Matbench Discovery's geometry-optimisation
  protocol on 10,000 WBM structures, drawn with a fixed seed from those whose
  elements are all in the model's table:
  - Relaxation: FIRE on a FrechetCellFilter to 0.05 eV/Å, for at most 500 steps,
    in float64, for this work, the authors' EnvTok model, MACE-OMAT-0 and
    MACE-MPA-0.
  - Scoring: the normalised RMSD to the DFT-relaxed structure, from
    StructureMatcher. A pair that does not match counts as 1 in the mean.
  - MACE-MPA-0 reproduces its published RMSD on the same structures: 0.0850
    against 0.0854, with a median per-structure difference of 4e-8.
  - Mean RMSD: 0.089 (this work), 0.092 (authors) and 0.081 (MACE-OMAT-0).
    Most of each mean comes from the 7 % of structures that do not match, and
    496 of those match in no model.
  - Only the RMSD is reported. The F1 score for stability is out of scope.
  - The OMat24-trained models are compared with DFT that uses the Materials
    Project's settings, so part of the gap is the difference between the two
    levels of theory.
- **Li-ion diffusion in solid electrolytes** (`envtok-mace diffusion`,
  `scripts/nscc/diffusion_8gpu.pbs`, `results/diffusion/`, figure `li_diffusion`).
  Five electrolytes (LGPS, Li7P3S11, β-Li3PS4, Li6PS5Cl, Li3InCl6) were run
  for this work, the authors' EnvTok model and MACE-OMAT-0:
  - Setup: the ordered Materials Project structures, as 351 to 540 atoms, at
    fixed cell. NVT MD at 600, 700, 800 and 900 K, with 10 ps equilibration and
    100 ps production.
  - Added after those runs: 500 and 550 K for Li7P3S11, so that this work has
    three usable temperatures, and 500, 550 and 650 K for β-Li3PS4, to cover the
    AIMD window.
  - Analysis: D comes from the Li mean squared displacement, and Ea from an
    Arrhenius fit over every usable temperature. σ at 300 K is extrapolated with
    Nernst–Einstein, taking a Haven ratio of 1.
  - All 75 runs finished without divergence.
  - Three electrolytes have AIMD values for the same structure; DOIs are in
    `literature_aimd.csv`. For LGPS the fitted Ea is 0.193 eV (this work),
    0.162 eV (authors) and 0.149 eV (MACE-OMAT-0), against 0.21 ± 0.04 eV. For
    Li7P3S11 it is 0.166, 0.174 and 0.171 eV, against 0.187 eV.
  - For β-Li3PS4, AIMD gives 0.49 ± 0.02 eV from 500 to 650 K. Fitted over the
    same temperatures, the models give 0.257, 0.220 and 0.391 eV, and over 500
    to 900 K, 0.266, 0.214 and 0.286 eV.
  - This work's Li7P3S11 fit leaves out 800 and 900 K. There its PS4 and P2S7
    units rotate, while P stays on its sites with four S. The framework
    criterion, fixed before the runs (non-Li MSD above 1.5 Å²), counts this as
    flow.
  - The Li3InCl6 references used a C2/m cell, and the Materials Project cell is
    C2, so they are not comparable.
- **`envtok-mace export-mace`** writes a model as a mace-torch file, which works
  with `MACECalculator` and compiles for LAMMPS without this package. In float64
  it agrees with this package to 1e-13 eV/Å once the rounding of the ZBL exponents
  in a float32-trained model is removed, and to 7e-7 eV/Å with it.
- **Inference speed** (`bench-speed --mode infer`, `scripts/nscc/bench_infer.pbs`,
  `results/benchmarks/inference_*.jsonl`, figure `inference_speed`): the time and
  memory of one calculator call from 4 to 23,328 atoms, on an A100 with either
  backend and on 16 CPU cores, in float32 and float64.
- **Model card** (`docs/MODEL_CARD.md`): training data, level of theory, intended
  use, limits, accuracy and speed.
- **Physics tests:** extensivity under supercells, forces summing to zero,
  locality beyond the cutoff, and smoothness as a pair crosses the cutoff.

### Changed

- The training figure has five panels: loss, force, energy and stress MAE, and
  the learning rate. Each panel shows the training, validation and held-out
  test sets: the training loss comes from the step log, and the training- and
  test-set errors from the 27 archived checkpoints
  (`results/runs/envtok_8gpu/train_loss.csv`, `checkpoints.csv`). The
  validation loss and stress MAE were logged before but not drawn.

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
- With the cuEquivariance backend, the ASE calculator failed with an illegal
  memory access on any GPU but the first, because the kernels launch on the
  current GPU. It now makes its own GPU current around loading and every call.
  A benchmark worker that hits a device fault stops without writing a result, so
  no earlier number is affected. Its message now includes the CUDA error.
  - This was also the fault that v1.0.0 hit in one MD rank and two NEB ranks,
    which `docs/BENCHMARKS.md` had put down to a float64 kernel fault. In v1.0.0
    those runs were redone with e3nn, so its numbers stand.
- `envtok_mace.__version__` said 0.1.0; it now follows the release version.

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

[Unreleased]: https://github.com/E1582271-dotcom/MACE-Envtok/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/E1582271-dotcom/MACE-Envtok/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/E1582271-dotcom/MACE-Envtok/releases/tag/v1.0.0
[0.9.0]: https://github.com/E1582271-dotcom/MACE-Envtok/releases/tag/v0.9.0
