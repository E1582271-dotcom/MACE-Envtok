# Parity with the reference mace-torch implementation

The model in `src/envtok_mace/model/` is written independently of mace-torch.

- **e3nn primitives** (spherical harmonics, Wigner 3j symbols, `o3.Linear`,
  `o3.TensorProduct`, `o3.FullyConnectedTensorProduct`, `FullyConnectedNet`,
  `nn.Activation`) are used as black boxes.
- **Written here**:
  - the radial basis, Agnesi transform, cutoff and ZBL repulsion;
  - the generalised Clebsch-Gordan coupling tensors;
  - the symmetric contraction (product basis) and the density-normalised interaction blocks;
  - model assembly;
  - forces and stress by autograd.

mace-torch v0.3.14 is used only as an oracle in the tests below
(`tests/test_parity_mace.py`). Weights are copied into our model through an
explicit name mapping (`model/mace_compat.py`) that must cover every parameter.

## Results

Test structures: 40 frames from the OMat24 validation split, 10 to 104 atoms,
55 elements (`tests/fixtures/omat24_val_40.h5`). The table gives the maximum
absolute difference over all frames, atoms and components.

| Check | dtype | Energy (eV) | Forces (eV/Å) | Stress (eV/Å³) |
|---|---|---|---|---|
| Random initialisation, same weights (8 frames) | float64 | 3.6e-15 | 1.4e-15 | 5.6e-17 |
| Authors' released EnvTok model (40 frames) | float64 | 2.8e-14 | 3.4e-13 | 5.6e-16 |
| Authors' released EnvTok model (40 frames) | float32 | 3.1e-5 (9.5e-7 per atom) | 3.1e-5 | 2.2e-7 |

Measured with torch 2.7.1, e3nn 0.4.4 and mace-torch 0.3.14 on CPU (the
versions used for training). An earlier run with torch 2.5.1 gave the same
orders of magnitude.

- **Parameter count:** 7,033,764 in both implementations.
- **Coupling tensors:** the U matrices of the symmetric contraction equal mace-torch's
  exactly (difference 0.0) for output irreps 0e and 1o and correlation orders 1 to 3.
- **Constants of a published model:** these (Bessel frequencies, Agnesi parameters,
  covalent radii, U matrices, E0s) are loaded from the model file. They are stored
  at float32 precision there, and recomputing them in float64 instead changes
  energies by about 1e-5 eV.
- **float32 differences:** these come from summation order only, since the same
  weights agree to 1e-13 in float64.
- **cuEquivariance backend:** the same model with the heavy operations on
  cuEquivariance kernels matches the e3nn backend (energies, forces, stresses and
  parameter gradients to a relative 1e-9 on CPU, `tests/test_backends.py`); the
  GPU check is part of the 8-GPU pilot.

## Physical checks (`tests/test_model_physics.py`)

- Energy is invariant, forces and stress equivariant, under a random rotation of
  positions and cell.
- Translation invariance; permutation equivariance of forces.
- Forces equal central finite differences of the energy (relative error < 1e-5).
- Stress equals the finite-strain derivative of the energy divided by the volume.
- A batch of structures gives the same energies, forces and stresses as
  evaluating each structure on its own.
