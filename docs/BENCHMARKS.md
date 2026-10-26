# MatCalc, MD stability and Li-migration benchmarks

The three benchmarks that drive the model through ASE: relaxations, elastic
constants, phonons, molecular dynamics and nudged elastic bands. MatPES, a plain
energy/force/stress test, is described in [MATPES.md](MATPES.md).

## Common setup

- **Calculator.** `envtok_mace.eval.ase_calc.EnvTokCalculator` rebuilds the
  neighbour list on every call with the training radius graph and returns
  ASE-convention stress. It matches the batched pipeline to 1e-12, and its forces
  and stress equal finite-difference derivatives of the energy
  (`tests/test_ase_calc.py`).
- **Precision.** float64 throughout, as in the paper.
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
3. **cuEquivariance kernel fault.** cuequivariance-ops 0.8.1 occasionally faults
   in float64 (`tensor_product_uniform_1d`, illegal memory access).
   - It hit one MD rank and four NEB workers.
   - The runners now stop a worker on a CUDA error instead of marking its
     remaining work as failed.
   - The affected 10 MD runs and 412 hops were redone with the e3nn backend,
     which agrees with cuEquivariance to 1e-11.
4. **GPU memory.** A float64 e3nn worker can hold 20+ GB while relaxing phonon
   supercells. Eight, and even two, workers per 40 GB GPU ran out of memory, and
   matcalc again recorded NaN without an error. MatCalc now runs one worker per
   GPU with a retry after an out-of-memory error. Every calculator exception is
   written to a log, so no NaN is left without a cause.
5. **Wall-clock-dependent scoring.** The MD score rewards fast steps. Hence the
   speed-matched results above.
