"""Bridge to the reference mace-torch implementation.

Used to (1) load published mace-torch models (e.g. the authors' EnvTok model)
into this implementation and (2) build randomly initialised mace-torch models
with the same architecture for parity tests. mace-torch is never used for training.
"""

from __future__ import annotations

import numpy as np
import torch

from .mace import MACE, MACEConfig


def config_from_mace(ref) -> MACEConfig:
    radial = ref.radial_embedding
    sc = ref.products[0].symmetric_contractions.contractions[0]
    readout = ref.readouts[-1]
    return MACEConfig(
        r_max=float(ref.r_max),
        num_bessel=len(radial.bessel_fn.bessel_weights),
        cutoff_p=int(radial.cutoff_fn.p),
        max_ell=ref.spherical_harmonics.irreps_out.lmax,
        num_channels=ref.node_embedding.linear.irreps_out.count((0, 1)),
        max_L=ref.products[0].linear.irreps_out.lmax,
        correlation=int(sc.correlation),
        num_interactions=int(ref.num_interactions),
        radial_mlp=tuple(ref.interactions[0].conv_tp_weights.hs[1:-1]),
        readout_hidden=readout.linear_1.irreps_out.dim,
        agnesi=hasattr(radial, "distance_transform"),
        pair_repulsion=hasattr(ref, "pair_repulsion"),
        z_table=tuple(int(z) for z in ref.atomic_numbers),
        e0s={
            int(z): float(e)
            for z, e in zip(ref.atomic_numbers, torch.atleast_2d(ref.atomic_energies_fn.atomic_energies)[0])
        },
        scale=float(torch.atleast_1d(ref.scale_shift.scale)[0]) if hasattr(ref, "scale_shift") else 1.0,
        shift=float(torch.atleast_1d(ref.scale_shift.shift)[0]) if hasattr(ref, "scale_shift") else 0.0,
    )


def _weight_map(model: MACE) -> dict[str, str]:
    """Our parameter name -> mace-torch parameter name."""
    cfg = model.config
    m = {"node_embedding.weight": "node_embedding.linear.weight"}
    for t in range(cfg.num_interactions):
        p = f"interactions.{t}."
        for name in ("linear_up.weight", "linear.weight", "skip_tp.weight", "density_fn.layer0.weight"):
            m[p + name] = p + name
        for i in range(len(cfg.radial_mlp) + 1):
            m[p + f"conv_tp_weights.layer{i}.weight"] = p + f"conv_tp_weights.layer{i}.weight"
        ours = model.products[t].symmetric_contraction
        for k in range(len(ours.out_irreps)):
            ref_c = f"products.{t}.symmetric_contractions.contractions.{k}."
            for nu in range(1, cfg.correlation + 1):
                idx = k * cfg.correlation + nu - 1
                theirs = "weights_max" if nu == cfg.correlation else f"weights.{cfg.correlation - 1 - nu}"
                m[f"products.{t}.symmetric_contraction.weights.{idx}"] = ref_c + theirs
        m[f"products.{t}.linear.weight"] = f"products.{t}.linear.weight"
    last = cfg.num_interactions - 1
    for t in range(last):
        m[f"readouts.{t}.weight"] = f"readouts.{t}.linear.weight"
    m[f"readouts.{last}.linear_1.weight"] = f"readouts.{last}.linear_1.weight"
    m[f"readouts.{last}.linear_2.weight"] = f"readouts.{last}.linear_2.weight"
    return m


def _buffer_map(model: MACE, ref) -> dict[str, str]:
    """Our constant buffers -> mace-torch buffers. A published model stores its
    constants (basis frequencies, Clebsch-Gordan tensors, ...) at its own precision;
    copying them makes the loaded model an exact replica."""
    cfg = model.config
    m = {
        "e0": "atomic_energies_fn.atomic_energies",
        "radial.frequencies": "radial_embedding.bessel_fn.bessel_weights",
        "radial.prefactor": "radial_embedding.bessel_fn.prefactor",
        "radial.r_max": "radial_embedding.cutoff_fn.r_max",
    }
    if cfg.agnesi:
        for ours, theirs in (
            ("agnesi_a", "a"),
            ("agnesi_q", "q"),
            ("agnesi_p", "p"),
            ("covalent_radii", "covalent_radii"),
        ):
            m[f"radial.{ours}"] = f"radial_embedding.distance_transform.{theirs}"
    if cfg.pair_repulsion:
        for name in ("c", "covalent_radii", "a_exp", "a_prefactor"):
            m[f"zbl.{name}"] = f"pair_repulsion_fn.{name}"
    if hasattr(ref, "scale_shift"):
        m["scale"], m["shift"] = "scale_shift.scale", "scale_shift.shift"
    for t in range(cfg.num_interactions):
        sc = model.products[t].symmetric_contraction
        for k in range(len(sc.out_irreps)):
            for nu in range(1, cfg.correlation + 1):
                m[f"products.{t}.symmetric_contraction.U_{k}_{nu}"] = (
                    f"products.{t}.symmetric_contractions.contractions.{k}.U_matrix_{nu}"
                )
    return m


def load_mace_weights(model: MACE, ref, copy_constants: bool = True) -> MACE:
    ref_params = dict(ref.named_parameters())
    mapping = _weight_map(model)
    ours = dict(model.named_parameters())
    unmapped = set(ours) - set(mapping)
    if unmapped:
        raise KeyError(f"parameters without a mace-torch counterpart: {sorted(unmapped)}")
    with torch.no_grad():
        for name, ref_name in mapping.items():
            src = ref_params[ref_name]
            if ours[name].shape != src.shape:
                raise ValueError(f"{name}: shape {tuple(ours[name].shape)} vs {ref_name} {tuple(src.shape)}")
            ours[name].copy_(src)
        if copy_constants:
            our_bufs, ref_bufs = dict(model.named_buffers()), dict(ref.named_buffers())
            z_used = model.z_table.long()
            for name, ref_name in _buffer_map(model, ref).items():
                dst, src = our_bufs[name], ref_bufs[ref_name]
                src = src.reshape(dst.shape)  # e.g. U for L=0 has no m axis in mace; e0 is [1, Z]
                a, b = dst.double(), src.double()
                if name.endswith("covalent_radii"):
                    # tables indexed by Z: only the model's elements matter (ASE versions differ
                    # in placeholder values for superheavy elements)
                    a, b = a[z_used], b[z_used]
                if not torch.allclose(a, b, rtol=1e-5, atol=1e-7):
                    raise ValueError(f"constant {name} differs from {ref_name} beyond rounding")
                dst.copy_(src)
    return model


def from_mace(ref) -> MACE:
    """Build our MACE with the same architecture and weights as a mace-torch model."""
    model = MACE(config_from_mace(ref))
    load_mace_weights(model, ref)
    return model.to(next(ref.parameters()).dtype)


def build_reference_mace(cfg: MACEConfig, avg_num_neighbors: float = 40.0):
    """A randomly initialised mace-torch ScaleShiftMACE with the given architecture."""
    from e3nn import o3
    from mace import modules

    hidden = " + ".join(
        f"{cfg.num_channels}x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(cfg.max_L + 1)
    )
    return modules.ScaleShiftMACE(
        r_max=cfg.r_max,
        num_bessel=cfg.num_bessel,
        num_polynomial_cutoff=cfg.cutoff_p,
        max_ell=cfg.max_ell,
        interaction_cls=modules.interaction_classes["RealAgnosticDensityResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticDensityInteractionBlock"],
        num_interactions=cfg.num_interactions,
        num_elements=len(cfg.z_table),
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps(f"{cfg.readout_hidden}x0e"),
        atomic_energies=np.array([cfg.e0s[z] for z in cfg.z_table]),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=list(cfg.z_table),
        correlation=cfg.correlation,
        gate=torch.nn.functional.silu,
        pair_repulsion=cfg.pair_repulsion,
        distance_transform="Agnesi" if cfg.agnesi else "None",
        radial_MLP=list(cfg.radial_mlp),
        use_reduced_cg=False,
        atomic_inter_scale=cfg.scale,
        atomic_inter_shift=cfg.shift,
    )


def mace_input_dict(batch, ref) -> dict[str, torch.Tensor]:
    """Our Batch -> the dict mace-torch's forward expects (on the batch's device)."""
    device = batch.positions.device
    z_table = ref.atomic_numbers.to(device)
    lookup = torch.full((int(z_table.max()) + 1,), -1, dtype=torch.long, device=device)
    lookup[z_table] = torch.arange(len(z_table), device=device)
    species = lookup[batch.numbers]
    dtype = next(ref.parameters()).dtype
    sender = batch.edge_index[0]
    cell = batch.cell.to(dtype)
    unit_shifts = batch.unit_shifts.to(dtype)
    ptr = torch.zeros(batch.num_graphs + 1, dtype=torch.long, device=device)
    ptr[1:] = torch.cumsum(batch.natoms, 0)
    return {
        "positions": batch.positions.to(dtype).clone(),
        "node_attrs": torch.nn.functional.one_hot(species, len(z_table)).to(dtype),
        "edge_index": batch.edge_index,
        "shifts": torch.einsum("ei,eij->ej", unit_shifts, cell[batch.batch[sender]]),
        "unit_shifts": unit_shifts,
        "cell": cell.reshape(-1, 3),
        "batch": batch.batch,
        "ptr": ptr,
        "head": torch.zeros(batch.num_graphs, dtype=torch.long, device=device),
    }
