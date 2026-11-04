# Model card: MACE-Envtok 1.0.0

A MACE interatomic potential trained from random initialisation on the EnvTok subset
of OMat24. It predicts energies, forces and stresses at the PBE(+U) level of OMat24,
for periodic structures made of 68 elements. It reproduces the EnvTok model of Moerman
et al., *Atoms that Matter* ([arXiv:2610.14392](https://arxiv.org/abs/2610.14392)).

| | |
|---|---|
| Weights | `mace-envtok-1.0.0.pt` (35 MB), best epoch (656) of the training run |
| SHA-256 | `f00b17a5ac50d8805a4be2271fd88d046c8ffa1c76a5c61ef673bf3b10e01e3d` |
| Elements | Z = 1–56 and 72–83; no lanthanides or actinides |
| Cutoff | 6 Å, with two interaction layers |
| Parameters | 7,033,764 |
| License | Weights CC BY 4.0, like OMat24; code MIT |

## Intended use

Research on inorganic crystals and other periodic materials made of these elements:
relaxations, molecular dynamics and screening, as a fast approximation to be checked
with DFT where the answer matters. It runs on CPU or GPU, through the ASE calculator
in this repository, mace-torch's `MACECalculator`, or LAMMPS
([Using the model](#using-the-model)).

## Where not to use it

- **Lanthanides and actinides.** These elements are outside the model, so it cannot
  run on structures that contain them, such as the solid electrolyte LLZO (La).
- **Next to Materials Project energies.** OMat24 uses version 54 of the VASP PBE
  pseudopotentials, not the legacy Materials Project defaults, and its authors warn
  about mixing the two ([arXiv:2410.12771](https://arxiv.org/abs/2410.12771)). Energies
  from this model need correcting before they go on a Materials Project convex hull.
- **Molecules, clusters and very short bonds.** Only 54 of 68 homonuclear diatomic
  curves have one well, and H, B, C and F have wells deeper than any homonuclear bond.
  On MatPES, the force error on non-bulk structures is 2.2 times that on bulk ones.
- **Where long-range interactions matter.** The model has no explicit electrostatics or
  dispersion, and PBE(+U) has no dispersion correction. Noble-gas solids show the limit:
  Ar, Kr and Xe have 6, 66 and 40 atoms in the held-out OMat24 set, and the fitted bulk
  modulus of the Xe cells ranges from 0.7 to 5 GPa between three models.

## Training data

The EnvTok subset of the OMat24 training split: 795,721 structures and 6,566,253 atoms,
less than 1 % of the dataset. The authors chose it with EnvTok and published the ids
([Zenodo 10.5281/zenodo.23137852](https://doi.org/10.5281/zenodo.23137852)).

The subset is not distributed like OMat24 as a whole. Against the held-out frames, its
cells are smaller (8.3 atoms against 18.3) and its forces larger (11 % of atoms above
5 eV/Å, against 7.5 %). AIMD at 3,000 K is 2.4 % of its atoms, against 52 %.

The level of theory is PBE, with Hubbard U for oxides and fluorides of Co, Cr, Fe, Mn,
Mo, Ni, V and W.

## Architecture and training

The architecture is that of MACE-OMAT-0 medium:

- 128 channels, L_max 1, l_max 3, correlation order 3, two interaction layers;
- 10 Bessel functions with the Agnesi transform;
- a ZBL pair repulsion at short range.

Training used a Huber loss on energy, forces and stress with weights 1 / 10 / 10, and
AMSGrad with learning rate 1e-3 and EMA 0.999. The global batch was 1,024 structures,
in float32. It ran on 8 A100 GPUs for 22 h, or 176 GPU-hours, and stopped early at epoch
696. There is one run from one seed, so the model has no uncertainty estimate, and
differences from the authors' EnvTok model cannot be attributed to a cause.

## Accuracy

MACE-OMAT-0, trained on all of OMat24, is the reference. The held-out OMat24 frames were
used neither for training nor for choosing the epoch. The README gives the full tables
and the other subset models.

| Test | Metric | This work | MACE-OMAT-0 |
|---|---|---|---|
| OMat24 held-out (591,981 frames) | Energy MAE, meV/atom | 24.59 | 17.47 |
| | Force MAE, meV/Å | 109.4 | 84.5 |
| | Stress MAE, meV/Å³ | 2.879 | 2.577 |
| MatPES bulk (242,319) | Median force MAE, meV/Å | 110.1 | 83.6 |
| MatPES non-bulk (80,766) | Median force MAE, meV/Å | 240.4 | 158.9 |
| MatCalc-Bench | Formation energy MAE, meV/atom | 62.9 | 56.0 |
| | Median bulk modulus error | +1.4 % | +3.2 % |
| | Median shear modulus error | −4.9 % | +1.4 % |
| nebDFT2k (1,656 Li hops) | Barrier MAE, hosts without U, eV | 0.138 | 0.146 |
| | Barrier MAE, hosts with U, eV | 0.259 | 0.270 |
| phononDB-PBE (103 binaries) | κ_SRME at 300 K (lower is better) | 0.381 | 0.248 |
| | Highest phonon frequency MAE, THz | 0.62 | 0.31 |

Li-ion diffusion was run in five solid electrolytes. Three of them have AIMD
values in the same structure:

| Electrolyte | This work | MACE-OMAT-0 | AIMD |
|---|---|---|---|
| LGPS | 0.193 eV | 0.149 eV | 0.21 eV |
| Li7P3S11 | 0.166 eV | 0.171 eV | 0.187 eV |
| β-Li3PS4 | 0.266 eV | 0.286 eV | 0.49 eV (500–650 K) |

These activation energies are fitted over the usable runs between 500 and 900 K.
They are predictions, compared with AIMD rather than experiment. The README gives
the caveats.

Force error varies by element, from 29 meV/Å for Na to 281 meV/Å for C. Predicted Li
barriers in hosts with U run high (median +83 meV), as expected from the mismatch: the
DFT reference has no U and the training data do.

## Physical checks

| Check | This work | MACE-OMAT-0 |
|---|---|---|
| Homonuclear diatomics with one well, of 68 | 54 | 57 |
| WBM equation-of-state curves with one well, of 610 | 597 | 598 |
| NPT runs ramped to 3,000 K and 500 GPa that diverged, of 80 | 0 | 31 |
| Median NVE drift, meV/atom/ps, over 10 ps | about 1e-4 | about 1e-4 |

Tests also cover invariance to rotation, translation and permutation, forces and stress
against finite differences, extensivity, forces summing to zero, and smoothness at the
cutoff.

The NVE drift is about the noise of the fit, in both float32 and float64, so this says
only that energy is conserved within what the test can resolve. Some of those structures
heat up far above their starting temperature, in all three models tested. The reason has
not been established.

## Using the model

The weights file needs this package, and runs on CPU with the e3nn backend:

```python
import torch
from envtok_mace.eval.run_eval import load_model
from envtok_mace.eval.ase_calc import EnvTokCalculator

model = load_model("mace-envtok-1.0.0.pt", torch.device("cpu"), torch.float64)
# on a GPU with the cueq extra installed: backend="cueq" for the fused kernels
atoms.calc = EnvTokCalculator(model, device="cpu", dtype="float64")
```

`envtok-mace export-mace` writes a mace-torch file, which needs mace-torch but not this
package:

```bash
envtok-mace export-mace --src mace-envtok-1.0.0.pt --dst mace-envtok.model
python -m mace.cli.create_lammps_model mace-envtok.model   # TorchScript for LAMMPS
```

```python
from mace.calculators import MACECalculator

atoms.calc = MACECalculator(model_paths="mace-envtok.model", device="cpu", default_dtype="float64")
```

The file and this package agree as follows on 40 OMat24 frames:

- **In float32,** they differ only by summation order: at most 5e-5 eV/Å in the forces.
- **In float64,** they differ by at most 7e-7 eV/Å. The model was trained in float32
  and holds the exponents of the ZBL screening function rounded to float32, while
  mace-torch has them in its code. With those exponents exact, the difference is 1e-13 eV/Å.

The TorchScript model, compiled from the released weights and run on its own on the same
frames, gives the same differences as the float64 file. It has not been run inside LAMMPS.

## Inference speed

The time for one ASE calculator call on fcc copper, with energy, forces, stress and the
neighbour list. Copper has 78 edges per atom, twice the 38 of a sample of OMat24 frames,
so these times are on the slow side.

| Atoms | A100, cuEquivariance | A100, e3nn | 16 CPU cores, e3nn |
|---|---|---|---|
| 108 | 0.049 s | 0.037 s | 0.71 s |
| 1,372 | 0.095 s | 0.25 s | 9.1 s |
| 4,000 | 0.22 s | 0.74 s | 28 s |
| 16,384 | 1.03 s | out of memory | 144 s |

*All in float32. The full results are in
[results/benchmarks/](../results/benchmarks/inference_a100.jsonl), and the
[figure](../results/figures/inference_speed.svg) also shows float64 and memory.*

- **Backend on a GPU:** cuEquivariance is 2.6 to 3.3 times faster than e3nn at 1,372 and
  4,000 atoms, and uses about a fifth of the memory. Below about 250 atoms e3nn is
  faster. The first cuEquivariance call takes 36 to 38 s, against under 4 s for e3nn.
- **Largest system tried on a 40 GB A100:**
  - cuEquivariance fits 23,328 atoms in float32 (32 GB, the largest tried) and 8,788 in
    float64;
  - e3nn fits 4,000 atoms in float32 and 1,372 in float64.
- **Neighbour list:** building the graph, which starts on the CPU, takes about 30 % of a
  cuEquivariance call from 4,000 atoms up.
- **CPU:** about 7 to 9 ms per atom in float32 and 11 to 14 ms in float64, from 500 atoms
  up. These were 16 cores of a shared node, so the times are noisier than on the GPU. At
  8,788 atoms in float64 the process reached 115 GB, and the next size was killed at the
  job's 110 GB limit.

## Citation

See [CITATION.cff](../CITATION.cff). Please also cite the paper this model reproduces,
OMat24 and MACE.
