# Benchmark protocols

This page covers the benchmarks that drive the model through ASE: relaxations,
elastic constants, phonons, molecular dynamics and nudged elastic bands.
- **The paper's three benchmarks:** MatCalc-Bench, MD stability and nebDFT2k.
- **Added in v1.1.0:**
  - homonuclear diatomics, equations of state and NVE drift;
  - phonons and thermal conductivity;
  - WBM relaxations;
  - Li-ion diffusion in solid electrolytes.

MatPES, a plain energy/force/stress test, is described in [MATPES.md](MATPES.md).

## Common setup

- **Calculator.** `envtok_mace.eval.ase_calc.EnvTokCalculator` rebuilds the
  neighbour list on every call with the training radius graph and returns
  ASE-convention stress. It matches the batched pipeline to 1e-12, and its forces
  and stress equal finite-difference derivatives of the energy
  (`tests/test_ase_calc.py`).
- **Precision.** float64, as in the paper. The exceptions are Li diffusion, which
  runs in float32, and NVE drift, which runs in both precisions.
- **Libraries.** Pinned to the authors' benchmark environment (the `bench`
  extra): matcalc 0.5.1, phonopy 4.1.0 with phonors 0.2.1, mlip-arena 0.2.0 and
  litraj 0.1.
- **Comparison models.** Our trained model is compared with the authors'
  published results for their EnvTok, DIRECT, four random-subset models and
  MACE-OMAT-0.
- **Validation.** Every pipeline was checked twice before our model was scored:
  1. the authors' published per-structure results, rescored with our code, must
     give the paper's numbers;
  2. the authors' released model, run through our pipeline, must reproduce their
     per-structure results.

## MatCalc-Bench (`envtok-mace matcalc`)

| Task | Data | Quantity |
|---|---|---|
| Equilibrium | 474 WBM structures | formation energy against tabulated MatPES-PBE elemental references; fingerprint distance to the DFT structure |
| Elasticity | 3,127 MP binaries | bulk and shear modulus (VRH), relaxed-ion, 6 strains per axis |
| Phonon | 1,050 Alexandria binaries | heat capacity C_V at 300 K |
| Softening | 480 WBM high-energy states | 1 − slope of predicted vs DFT forces |

Structures with elements outside the model's table are dropped. Moduli and heat
capacities are scored as signed percentage errors; failed calculations and
non-positive predictions are excluded, as in the paper.

**Validation.** Scoring the authors' CSVs with our code gives the paper's
formation-energy MAE of 62.7 meV/atom and shear-modulus median of −5.8 %. Their
EnvTok model, run through our pipeline on 40 structures per task, reproduces
their published values:

| Quantity | Largest relative difference |
|---|---|
| K | 1e-13 |
| G | 4e-13 |
| C_V | 2e-10 |
| Softening scale | 2e-10 |
| Formation energy | 2e-4 (median) |

Formation energies agree less closely because each relaxation starts from a
randomly perturbed structure.

**Results** (median; formation energy also as the paper's MAE).

| Model | E_form MAE (meV/atom) | E_form median | K signed | G signed | C_V signed | Softening |
|---|---|---|---|---|---|---|
| **Ours** | 62.9 | 25.3 | **+1.41 %** | −4.88 % | +0.41 % | 3.63 % |
| Authors, EnvTok | 62.7 | 28.0 | +1.47 % | −5.77 % | +0.52 % | 3.77 % |
| Authors, DIRECT | 74.3 | 35.8 | +2.01 % | −8.03 % | +0.44 % | 3.91 % |
| Authors, random (4 seeds pooled) | 65.5 | 27.3 | +2.73 % | −5.62 % | +0.38 % | 3.32 % |
| MACE-OMAT-0 | 56.0 | 18.8 | +3.17 % | +1.42 % | +0.13 % | 2.40 % |

The paper notes that subset training biases G low. Our model has the smallest G
bias among the subset models. The mean G error (+16 %) is larger than for the
authors' EnvTok model (+3.6 %): a handful of structures with very large positive
errors, beyond the 95th percentile, dominate the mean. Our 95th percentile
(67.7 %) is lower than theirs (73.0 %). Full distributions are in
`results/matcalc/matcalc_summary.json`.

## MD stability (`envtok-mace stability`)

- **Runs.** 80 NPT compression runs on RM24 random mixtures (77–1,500 atoms),
  using mlip-arena's own MD task: 300 → 3,000 K and 0 → 500 GPa over 10 ps.
- **Wall-clock budget.** Each run gets three attempts of 600 s wall time.
- **Outcomes.** Each run is classified as survived, physically diverged
  (singular NPT cell, or temperature above 1e5 K), OOM, other failure or
  truncated, in the paper's order of precedence.

**Validation.** On the authors' published runs, the classifier gives the
paper's numbers:

| Model | Mean completion | Divergences |
|---|---|---|
| EnvTok | 92.1 % | 6 / 80 |
| MACE-OMAT-0 | 75.4 % | 31 / 80 |
| Random seeds | — | 85 / 320 |

Their published DIRECT runs give 9 divergences rather than the paper's 1; the
authors note that the DIRECT values depend on the DIRECT model used.

**Speed.** The completion score depends on how fast a step is, because slow runs
are truncated by the wall-clock budget. Against the authors' runs on the same
structures, our runs were faster:

| Our backend | Runs | Median speed-up over the authors |
|---|---|---|
| cuEquivariance | 70 | 2.5× |
| e3nn | 10 | 1.35× |

The per-step medians are in `results/stability/speed_vs_authors.json`.

So every model is also scored with each run cut where the authors' EnvTok setup
would have run out of time ("at their speed"). Scoring the authors' runs this way
leaves them unchanged.

**Results.**

| Model | Mean completion | Finished | Physically diverged | OOM | Truncated |
|---|---|---|---|---|---|
| **Ours, at the authors' speed** | **96.1 %** | 61 | **0** | 0 | 19 |
| Ours, as measured | 99.9 % | 79 | 0 | 0 | 1 |
| Authors, EnvTok | 92.1 % | 54 | 6 | 1 | 19 |
| Authors, DIRECT | 92.9 % | 51 | 9 | 2 | 18 |
| Authors, random (4 seeds pooled) | 81.1 % | 159 / 320 | 85 / 320 | 8 / 320 | 67 / 320 |
| MACE-OMAT-0 | 75.4 % | 26 | 31 | 9 | 14 |

No run of our model diverged. This is one training run. The authors' four random
seeds range from 14 to 33 divergences, so stability under this protocol varies a
lot from run to run.

## LiTraj nebDFT2k (`envtok-mace nebdft2k`)

**Protocol.** 1,656 Li-vacancy hops, following the paper's variant of the LiTraj
recipe:

1. relax both end-points (FIRE, fmax 0.05 eV/Å, up to 10,000 steps);
2. relax the climbing-image band (k = 5 eV/Å², improved tangent, same limits);
3. take the barrier as max − min energy along the band.

A hop is scored if the largest force component is below 0.1 eV/Å. Hosts with
Hubbard-U elements are reported separately, since the DFT reference has no U.

**Validation.** Rescoring the authors' hops gives:

- 0.124 eV MAE without U for EnvTok;
- 0.148 ± 0.037 eV for the random seeds.

Both match the paper. On the hop that was checked, their model run through our
pipeline reproduces the barrier (0.48480 eV) and the optimiser step counts
(47 / 50 / 62).

**Results** (converged hops, eV).

| Model | Without U: MAE | RMSE | median abs | With U: MAE |
|---|---|---|---|---|
| **Ours** | 0.138 | 0.440 | 0.053 | **0.259** |
| Authors, EnvTok | 0.124 | 0.372 | 0.050 | 0.269 |
| Authors, DIRECT | 0.116 | 0.325 | 0.059 | 0.316 |
| Authors, random (4 seeds) | 0.148 ± 0.037 | 0.795 | 0.051 | 0.264 |
| MACE-OMAT-0 | 0.146 | 0.728 | 0.038 | 0.270 |

Our model lies within the spread of the random seeds without U and is slightly
ahead with U.

## Homonuclear diatomics (`envtok-mace diatomics`)

**Protocol.** This is MLIP Arena's homonuclear task for each of the 68 elements:
- **Curve:** two atoms in vacuum, from 0.9 × the covalent radius to 3.1 × the
  van der Waals radius, in 0.01 Å steps.
- **Code:** the arena's own function is called without prefect, and the arena's
  smoothness metrics are computed from each curve.
- **Energy zero:** the arena puts it at the largest distance. For light elements
  that distance lies inside the 6 Å cutoff, so the curves are referenced to two
  isolated atoms computed with the same model.

**Results.** These are counts over 68 elements.

| Model | One well | Repulsive at close range | Wells deeper than 12 eV |
|---|---|---|---|
| **Ours** | 54 | 66 | H, B, C, F |
| Authors, EnvTok | 39 | 65 | N, F, Si, P, Cr, Mn |
| MACE-OMAT-0 | 57 | 67 | Ar |

The strongest homonuclear bond, N₂, is 9.8 eV, so a well deeper than 12 eV is a
curve that collapses at short range. OMat24 holds crystals, not molecules, so
these curves test the models far from their training data.

## Equation of state (`envtok-mace eos`)

**Protocol.** These are MLIP Arena's `eos_bulk` steps on its 1,000 WBM
structures. The 610 structures whose elements are all in the model's table were
run:
1. relax the positions (FIRE, fmax 0.1 eV/Å), then the cell as well
   (FrechetCellFilter, fmax 0.05 eV/Å);
2. scale the cell to 21 volumes within ±20 %, and relax the positions at each one;
3. fit a Birch–Murnaghan equation to get the bulk modulus.

**Results.** Smoothness is scored with the arena's metrics. On the 610
structures:

| Model | One well | No fit | Median \|ΔK\| against MACE-OMAT-0 |
|---|---|---|---|
| **Ours** | 597 | 1 | 3.7 GPa (606 structures) |
| Authors, EnvTok | 593 | 7 | 3.7 GPa (602 structures) |
| MACE-OMAT-0 | 598 | 3 | — |

The set has no DFT moduli, so the bulk modulus is compared with MACE-OMAT-0
rather than with DFT.

## NVE drift (`envtok-mace nve`)

**Protocol.**
- **Structures:** a fixed random sample of 20 structures from the WBM set above,
  each repeated to at least 64 atoms.
- **Velocities:** drawn at twice the target temperature, because about half of
  the kinetic energy goes into potential energy in the first steps.
- **MD:** the arena's velocity Verlet task, with 2 fs steps (0.5 fs with
  hydrogen), for 10 ps.
- **Runs:** starts at 300 and 1,000 K, in float32 and float64.
- **Drift:** the slope of a line through the total energy per atom.

**Results.** None of the 240 runs failed. The median |drift| is about 1e-4
meV/atom/ps for every model, start temperature and precision, near the
resolution of the fit, so the test cannot rank the models. Runs are reported at
their measured temperature, because some structures heat far above their start
in all three models. C27Rb27Ti81 goes from a nominal 300 K to 1,500–1,800 K, and
it has the largest drift, about 0.02 meV/atom/ps. Relaxing first would not help:
on the three structures checked, relaxation releases almost no energy.

## Phonons and thermal conductivity (`envtok-mace kappa`)

**Protocol.** This is Matbench Discovery's κ_SRME task on its 103 phononDB-PBE
binaries, with phono3py 4.1.0:
1. relax with FIRE and a FrechetCellFilter without tilt, keeping the symmetry,
   to 1e-4 eV/Å in at most 300 steps;
2. compute second- and third-order force constants from 0.01 Å displacements;
3. solve the Wigner transport equation at 300 K, on the reference's q mesh.

A structure that changes its space group or has imaginary modes gets no
conductivity and the maximum error of 2, as upstream. Forces run on a GPU
(`kappa_forces.pbs`), and the transport equation on CPU cores
(`kappa_conductivity.pbs`). The reference also holds the DFT frequencies, so the
same run gives each structure's highest frequency, and its heat capacity, free
energy and entropy at 300 K. The scoring rules are Matbench Discovery's,
implemented in `bench/kappa.py`.

**Validation.** Run through this pipeline, MACE-MPA-0 gives its published
numbers:

| Quantity | This pipeline | Published |
|---|---|---|
| κ_SRME | 0.4131 | 0.4123 |
| κ_SRE | 0.2048 | 0.2048 |
| Failure rate | 0 | 0 |

**Results.**

| Model | κ_SRME | κ_SRE | Scored 2 | ω_max MAE (THz) | C_V MAE (J/K/mol) | S MAE (J/K/mol) |
|---|---|---|---|---|---|---|
| **Ours** | 0.381 | 0.201 | 1 (PbS) | 0.62 | 0.12 | 0.64 |
| Authors, EnvTok | 0.405 | 0.222 | 0 | 0.68 | 0.15 | 0.81 |
| MACE-OMAT-0 | 0.248 | 0.127 | 0 | 0.31 | 0.066 | 0.30 |
| MACE-MPA-0 | 0.413 | 0.205 | 0 | 0.83 | 0.28 | 0.84 |

Our model's one failure is PbS, which has imaginary modes.

## Li-ion diffusion in solid electrolytes (`envtok-mace diffusion`)

**Protocol.**
- **Structures:** five electrolytes in the ordered structure the Materials
  Project computed: LGPS, Li7P3S11, β-Li3PS4, Li6PS5Cl and Li3InCl6. Each cell is
  repeated to at least 200 atoms, so 351–540 atoms. The cell is kept at its PBE
  volume, the same for every model, as in AIMD studies.
- **Not included:**
  - Li3YCl6, because its Y sites are partly occupied, so no database has it in
    the experimental structure;
  - LLZO, because it contains La, which is outside the model.
- **MD:** a Nosé–Hoover chain with 2 fs steps, in float32. 10 ps of
  equilibration, then 100 ps whose positions are saved every 50 fs.
- **Temperatures:** 600, 700, 800 and 900 K. Added after these first 60 runs:
  500 and 550 K for Li7P3S11, and 500, 550 and 650 K for β-Li3PS4.
- **D:** the slope of the Li mean squared displacement from 5 to 50 ps of lag,
  averaged over time origins, after removing the drift of the framework's centre.
  Its spread comes from four 25 ps blocks.
- **Ea and σ:** a line through ln D against 1/T over every usable temperature
  gives Ea and D at 300 K. Nernst–Einstein with an assumed Haven ratio of 1 then
  gives σ at 300 K.
- **Flags, fixed before the runs:**
  - a run is flagged rather than fitted when the Li MSD is below 3 Å² or the
    framework's MSD is above 1.5 Å²;
  - it stops as diverged if two atoms come within 0.6 Å or the temperature
    passes twice the target.

**Validation.** No published MLIP diffusion results for these models exist to
reproduce. The MSD analysis gives back the known D of random walks
(`tests/test_diffusion.py`), and the thermostat holds every run within 0.3 % of
its temperature. The comparison is with AIMD in the same structure, where it
exists (`results/diffusion/literature_aimd.csv`).

**Results.** All 75 runs finished without divergence. The tables are in
`results/diffusion/`, and the README gives Ea against AIMD.

**Caveats.**
- **Li7P3S11, this work:** at 800 and 900 K its PS4 and P2S7 units rotate, while
  every P keeps its four S. The framework criterion counts this as flow (non-Li
  MSD 1.56 and 2.43 Å²), which leaves 500–700 K for the fit. Fitting all six
  temperatures anyway, as a sensitivity check only, gives the same 0.166 eV.
- **β-Li3PS4:** the AIMD reference covers 500–650 K. Over that window the models
  give 0.257, 0.220 and 0.391 eV, against 0.49 ± 0.02 eV. With only four
  temperatures, dropping the 500 K point moves these fits by up to 0.05 eV.
- **Li3InCl6:** the AIMD references used a C2/m cell, while the Materials Project
  cell is C2, so they are not comparable.
- **Single runs:** each temperature is one 100 ps run. The room-temperature
  conductivity is an extrapolation from the lowest fitted temperature, 500 or
  600 K, down to 300 K.

## Problems found on the way

1. **phonopy / phonors mismatch.** phonopy 4.1.0 calls a function that phonors
   0.6 renamed. Every phonon calculation then fails, and matcalc records it as
   NaN without an error. phonors is pinned to 0.2.1, the authors' version.
2. **Element lookup.** The atomic-number lookup was only as long as the largest
   supported element (Z = 83). On a GPU, Z ≥ 84 indexed past its end, which is a
   device-side assertion that leaves the process unable to use CUDA. In
   Softening, whose structures cannot be filtered before they run, one actinide
   compound made every later structure on that worker fail. The table now covers
   every element, and out-of-table elements raise a normal error.
3. **cuEquivariance on a node's second GPU.** cuEquivariance launches its kernels
   on the current GPU, not on the GPU that holds the tensors. A rank that PBS
   placed on the second GPU of a node therefore failed with an illegal memory
   access at its first call.
   - It hit the MD rank and the two NEB ranks that shared a node with another
     rank. That was the ten MD runs of one rank and the 412 hops of eight
     workers. Every rank on a node's first GPU ran normally.
   - The affected runs were redone with the e3nn backend, which agrees with
     cuEquivariance to 1e-11.
   - The runners now stop a worker on a CUDA error instead of marking its
     remaining work as failed.
   - The cause was found later, in the Li diffusion runs, and earlier versions
     of this page blamed a float64 kernel fault. The calculator now makes its
     own GPU current (see the changelog).
4. **GPU memory.** A float64 e3nn worker can hold 20+ GB while relaxing phonon
   supercells. Eight, and even two, workers per 40 GB GPU ran out of memory, and
   matcalc again recorded NaN without an error. MatCalc now runs one worker per
   GPU with a retry after an out-of-memory error. Every calculator exception is
   written to a log, so no NaN is left without a cause.
5. **Wall-clock-dependent scoring.** The MD score rewards fast steps. Hence the
   speed-matched results above.
6. **κ force stage out of memory.** Four float64 workers sharing one A100 ran
   out of memory on 6–7 dense light-element cells per model, such as BN, BeO
   and AlN. The force stage now retries those records, and the run used one
   worker per GPU.
7. **Wyckoff letters depend on the setting.** In Pnma, 4a and 4b differ only by
   the choice of origin, so the letter spglib reports depends on how the cell is
   placed. An early comparison used the letters and called the β-Li3PS4 AIMD
   reference another Li ordering. The site symmetry (−1 against .m.) shows it
   is the same ordering.
