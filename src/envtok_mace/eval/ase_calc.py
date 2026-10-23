"""ASE calculator around our MACE implementation, for relaxations, elasticity,
phonons and MD driven by ASE-based tools (matcalc, phonopy, MLIP-Arena).

The neighbour list is rebuilt on every call with the same radius graph the
training data used, so a structure gives the same edges here as in training.
Stress follows ASE: Voigt order (xx, yy, zz, yz, xz, xy), eV/A^3, and the sign
of (1/V) dE/de, positive under tension.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from ..data.neighbors import radius_graph

_VOIGT_ROWS = (0, 1, 2, 1, 0, 0)
_VOIGT_COLS = (0, 1, 2, 2, 2, 1)


class EnvTokCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress", "energies"]

    def __init__(self, model, device: str | torch.device = "cpu",
                 dtype: torch.dtype | str = torch.float64, backend: str | None = None, **kwargs):
        """``model``: a MACE instance, or a path accepted by ``eval.run_eval.load_model``
        (our checkpoints or a released mace-torch .model)."""
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
            from .run_eval import load_model

            model = load_model(str(model), self.device, self.dtype, backend)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.r_max = float(self.model.config.r_max)

    def _batch(self, atoms) -> SimpleNamespace:
        pos = atoms.get_positions()
        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        edge_index, shifts = radius_graph(pos, cell, atoms.get_pbc(), self.r_max)
        t = lambda a, dt=self.dtype: torch.as_tensor(a, dtype=dt, device=self.device)  # noqa: E731
        n = len(atoms)
        return SimpleNamespace(
            positions=t(pos), numbers=t(atoms.get_atomic_numbers(), torch.long),
            batch=torch.zeros(n, dtype=torch.long, device=self.device),
            edge_index=t(edge_index, torch.long), unit_shifts=t(shifts),
            cell=t(cell).unsqueeze(0), num_graphs=1,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # Stress costs one extra gradient, so it is always returned for a periodic
        # cell: ASE optimisers with a cell filter ask for it right after the forces.
        want_stress = bool(self.atoms.pbc.all()) and self.atoms.cell.rank == 3
        out = self.model(self._batch(self.atoms), compute_force=True,
                         compute_stress=want_stress, training=False)
        energy = float(out["energy"].detach().item())
        self.results = {
            "energy": energy, "free_energy": energy,
            "forces": out["forces"].detach().cpu().numpy().astype(np.float64),
            "energies": out["node_energy"].detach().cpu().numpy().astype(np.float64),
        }
        if want_stress:
            s = out["stress"].detach()[0]
            self.results["stress"] = s[_VOIGT_ROWS, _VOIGT_COLS].cpu().numpy().astype(np.float64)
