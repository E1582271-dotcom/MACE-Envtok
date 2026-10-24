"""Score a model on the MatPES test set the way the paper's tables are built.

Two steps, as in the authors' benchmark. ``--run`` scores one model and writes
``per_structure.npz`` and ``per_atom_forces.npz`` into an output directory, in the
same layout and with the same quantities as their published files, so the two can
be compared array by array. ``--tables`` then turns several such directories into
the paper's estimator tables.

Why the per-structure errors are summarised with medians and clipped means: on
MatPES the non-bulk domain contains structures where a subset model's force error
runs to thousands of meV/A, and a plain mean over the domain is then a statement
about a handful of structures. The estimators here are the authors' --
TRIM_FRACTION 0.05 of the weight off the upper tail, the winsorised mean clipped
at the 0.99 quantile of all models pooled (the same absolute cap for every model,
so no model gains from a longer tail), and the pooled mean kept alongside.

The bulk / non-bulk / overcompressed labels come from the geometric descriptors
in the authors' published per-structure file (see ``paper_domains``), which
depend on the reference geometry alone and on no model's errors. Using their
descriptors is what makes the bins identical to the paper's.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..data.batch import collate
from ..data.frames import Frames, _ranges
from ..data.matpes import KBAR_TO_EV_PER_A3, STRESS_ORDERS
from ..data.samplers import eval_shards

# Per-structure quantities and the weight each is pooled with (atom-weighted for
# force errors, so a pooled mean over a domain equals the mean over its atoms).
QUANTITIES: tuple[tuple[str, str | None, str], ...] = (
    ("f_mae_comp", "natoms", "meV/A"),
    ("f_rmse_comp", "natoms", "meV/A"),
    ("f_max_norm", "natoms", "meV/A"),
    ("abs_d_energy_per_atom", None, "meV/atom"),
    ("s_mae_voigt6", None, "meV/A^3"),
    ("s_err_norm", None, "meV/A^3"),
)
DOMAIN_ORDER = ("all", "bulk", "non_bulk", "overcompressed")
TRIM_FRACTION = 0.05
WINSOR_QUANTILE = 0.99

# 3x3 -> ASE Voigt 6 (xx, yy, zz, yz, xz, xy)
_VOIGT_ROWS = (0, 1, 2, 1, 0, 0)
_VOIGT_COLS = (0, 1, 2, 2, 2, 1)


# --------------------------------------------------------------------- scoring
def score(model, frames: Frames, batches: list[np.ndarray], device: torch.device,
          dtype: torch.dtype) -> tuple[dict, dict]:
    """Per-structure and per-atom errors, in the frames' own order."""
    n, a = len(frames), int(frames.atom_ptr[-1])
    ps = {k: np.full(n, np.nan) for k, _, _ in QUANTITIES}
    d_force = np.zeros((a, 3), dtype=np.float32)
    for idx in batches:
        batch = collate(frames, idx).to(device=device, dtype=dtype)
        out = model(batch, compute_force=True, compute_stress=True, training=False)
        ng = batch.num_graphs
        g, na = batch.batch, batch.natoms.double()
        df = (out["forces"].detach() - batch.forces).double()
        zeros = torch.zeros(ng, dtype=torch.float64, device=device)
        f_abs = zeros.clone().index_add_(0, g, df.abs().sum(dim=1))
        f_sq = zeros.clone().index_add_(0, g, (df**2).sum(dim=1))
        de = (out["energy"].detach().double() - batch.energy.double()).abs() / na
        ds = out["stress"].detach().double() - batch.stress.double()
        s6 = ds[:, _VOIGT_ROWS, _VOIGT_COLS]

        rows = np.asarray(idx)
        ps["f_mae_comp"][rows] = (f_abs / (3 * na)).cpu().numpy()
        ps["f_rmse_comp"][rows] = (f_sq / (3 * na)).sqrt().cpu().numpy()
        ps["abs_d_energy_per_atom"][rows] = de.cpu().numpy()
        ps["s_mae_voigt6"][rows] = s6.abs().mean(dim=1).cpu().numpy()
        ps["s_err_norm"][rows] = ds.flatten(1).norm(dim=1).cpu().numpy()
        d_force[_ranges(frames.atom_ptr[rows], frames.atom_ptr[rows + 1])] = \
            df.float().cpu().numpy()

    # The largest per-atom error of a structure, from the per-atom layer, so the
    # two files cannot disagree. Structures that were not scored stay NaN.
    err_norm = np.linalg.norm(d_force, axis=1).astype(np.float32)
    ps["f_max_norm"] = np.maximum.reduceat(err_norm.astype(np.float64), frames.atom_ptr[:-1])
    ps["f_max_norm"][np.isnan(ps["f_mae_comp"])] = np.nan

    from ase.data import chemical_symbols

    symbols = np.array(chemical_symbols)
    elements, n_elements = [], np.empty(n, dtype=np.int64)
    for i in range(n):
        u = np.unique(frames.numbers[frames.atom_ptr[i]:frames.atom_ptr[i + 1]])
        elements.append("".join(sorted(symbols[u])))
        n_elements[i] = len(u)

    per_structure = {
        "index": np.arange(n, dtype=np.int64), "sid": frames.sid.astype(str),
        "natoms": frames.natoms.astype(np.int64), "n_elements": n_elements,
        "elements": np.array(elements), **ps,
        "e_finite": np.isfinite(ps["abs_d_energy_per_atom"]),
        "f_finite": np.isfinite(ps["f_mae_comp"]),
        "s_finite": np.isfinite(ps["s_mae_voigt6"]),
    }
    per_atom = {
        "d_force": d_force,
        "err_norm": err_norm,
        "ref_norm": np.linalg.norm(frames.forces, axis=1).astype(np.float32),
        "Z": frames.numbers.astype(np.int16),
        "frame": np.repeat(np.arange(n, dtype=np.int32), frames.natoms),
        "struct_offset": frames.atom_ptr.astype(np.int64),
    }
    return per_structure, per_atom


# --------------------------------------------------- weighted estimators (theirs)
def wquantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile; ``v`` may contain +inf."""
    ok = w > 0
    if not ok.any():
        return float("nan")
    v, w = np.asarray(v, float)[ok], np.asarray(w, float)[ok]
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    return float(v[o][min(int(np.searchsorted(cw, q * cw[-1])), len(v) - 1)])


def wmean(v: np.ndarray, w: np.ndarray) -> float:
    ok = w > 0
    return float(np.average(np.asarray(v, float)[ok], weights=np.asarray(w, float)[ok])) \
        if ok.any() else float("nan")


def wtrimmed_mean(v: np.ndarray, w: np.ndarray, frac: float = TRIM_FRACTION) -> float:
    """Mean after dropping the worst ``frac`` of the weight (upper tail only)."""
    ok = w > 0
    if not ok.any():
        return float("nan")
    v, w = np.asarray(v, float)[ok], np.asarray(w, float)[ok]
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    keep = np.cumsum(w) <= (1.0 - frac) * w.sum()
    if not keep.any():
        keep[0] = True
    return float(np.average(v[keep], weights=w[keep]))


def wwinsor_mean(v: np.ndarray, w: np.ndarray, cap: float) -> float:
    return wmean(np.clip(np.asarray(v, float), -np.inf, cap), w)


def wgeomean(v: np.ndarray, w: np.ndarray, floor: float = 1e-9) -> float:
    ok = (w > 0) & np.isfinite(v)
    if not ok.any():
        return float("nan")
    return float(np.exp(np.average(np.log(np.maximum(np.asarray(v, float)[ok], floor)),
                                   weights=np.asarray(w, float)[ok])))


def pooled_cap(values: list[np.ndarray], weights: list[np.ndarray],
               q: float = WINSOR_QUANTILE) -> float:
    """The common clipping cap: quantile ``q`` of every model's values pooled."""
    v = np.concatenate([np.asarray(x, float) for x in values])
    w = np.concatenate([np.asarray(x, float) for x in weights])
    fin = np.isfinite(v)
    return wquantile(v[fin], w[fin], q) if fin.any() else float("nan")


def worst_fill(v: np.ndarray) -> np.ndarray:
    """Replace non-finite errors by the worst finite one, never by a small number."""
    v = np.asarray(v, float).copy()
    bad = ~np.isfinite(v)
    if bad.any():
        v[bad] = v[~bad].max() if (~bad).any() else np.inf
    return v


ESTIMATORS: tuple[tuple[str, object], ...] = (
    ("median", lambda v, w, cap: wquantile(v, w, 0.50)),
    ("q75", lambda v, w, cap: wquantile(v, w, 0.75)),
    ("q90", lambda v, w, cap: wquantile(v, w, 0.90)),
    ("winsor_mean", lambda v, w, cap: wwinsor_mean(v, w, cap)),
    ("trimmed_mean*", lambda v, w, cap: wtrimmed_mean(v, w)),
    ("geometric_mean", lambda v, w, cap: wgeomean(v, w)),
    ("pooled_mean_MAE", lambda v, w, cap: wmean(v, w)),
)


# ----------------------------------------------------------------------- tables
def _load_runs(dirs: list[str], labels: list[str]) -> dict[str, dict]:
    runs = {}
    for label, d in zip(labels, dirs):
        z = np.load(Path(d) / "per_structure.npz", allow_pickle=False)
        runs[label] = {k: z[k] for k in z.files}
    return runs


def paper_domains(run: dict) -> np.ndarray:
    """The domains of the paper's tables, from the descriptors in the authors' file.

    Their per-structure files store a domain from the evaluation-time detector
    (dimensionality < 3 or more than one fragment), but the published tables
    were built after relabelling with dimensionality alone ("dim"): 3,118
    structures move from non-bulk to bulk. Overcompressed cells keep their stored
    label. This reproduces their 242,319 / 80,766 / 11 split exactly.
    """
    stored = run["domain"].astype(str)
    return np.where(stored == "overcompressed", "overcompressed",
                    np.where(run["dim"] < 3, "non_bulk", "bulk"))


def align(runs: dict[str, dict], domain_from: str | None,
          rule: str = "paper") -> tuple[dict[str, dict], np.ndarray]:
    """Put every run on the sid order of the first one; return the domain labels."""
    labels = list(runs)
    order = runs[labels[0]]["sid"].astype(str)
    out = {}
    for label in labels:
        sid = runs[label]["sid"].astype(str)
        pos = {s: i for i, s in enumerate(sid)}
        missing = [s for s in order if s not in pos]
        if missing:
            raise ValueError(f"{label} is missing {len(missing)} sids, e.g. {missing[:3]}")
        take = np.array([pos[s] for s in order])
        out[label] = {k: v[take] if getattr(v, "shape", ()) == sid.shape else v
                      for k, v in runs[label].items()}
    src = domain_from or labels[0]
    if "domain" not in out[src]:
        raise ValueError(f"run {src} has no domain column; point --domain-from at the "
                         f"authors' per_structure.npz directory")
    if rule == "paper":
        return out, paper_domains(out[src])
    return out, out[src]["domain"].astype(str)


def tables(runs: dict[str, dict], domain: np.ndarray) -> dict:
    labels = list(runs)
    masks = {"all": np.ones(len(domain), bool)}
    for k in DOMAIN_ORDER[1:]:
        m = domain == k
        if m.any():
            masks[k] = m
    out = {"labels": labels, "n_structures": int(len(domain)), "metrics": {}}
    for key, wcol, unit in QUANTITIES:
        if not all(key in runs[L] for L in labels):
            continue
        out["metrics"][key] = {"unit": unit, "domains": {}}
        for dk, m in masks.items():
            vw = [(worst_fill(runs[L][key][m]),
                   runs[L][wcol][m].astype(float) if wcol else np.ones(int(m.sum())))
                  for L in labels]
            cap = pooled_cap([v for v, _ in vw], [w for _, w in vw])
            dom = {"n": int(m.sum()), "cap": cap, "estimators": {}}
            for name, fn in ESTIMATORS:
                dom["estimators"][name] = {L: fn(v, w, cap) for L, (v, w) in zip(labels, vw)}
            out["metrics"][key]["domains"][dk] = dom
    return out


def write_tables(result: dict, out_dir: str | Path, scale: float = 1e3) -> None:
    """One CSV per quantity and domain (values scaled to meV like the paper's)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = result["labels"]
    for key, meta in result["metrics"].items():
        for dk, dom in meta["domains"].items():
            path = out_dir / f"{key}_{dk}.csv"
            with path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow([f"# {key} over domain {dk}, in {meta['unit']}"])
                w.writerow([f"# {dom['n']} structures, winsor cap "
                            f"{scale * dom['cap']:.6g} {meta['unit']}"])
                w.writerow(["estimator"] + labels)
                for name, vals in dom["estimators"].items():
                    w.writerow([name] + [f"{scale * vals[L]:.6g}" for L in labels])
    (out_dir / "matpes_tables.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {len(result['metrics']) * len(DOMAIN_ORDER)} tables -> {out_dir}", flush=True)


# ------------------------------------------------------------- cross-check modes
def compare_per_atom(ours_dir: str | Path, theirs_dir: str | Path) -> dict:
    """Atom-by-atom agreement with the authors' own run of the same model."""
    ours_s = np.load(Path(ours_dir) / "per_structure.npz", allow_pickle=False)
    theirs_s = np.load(Path(theirs_dir) / "per_structure.npz", allow_pickle=False)
    ours_a = np.load(Path(ours_dir) / "per_atom_forces.npz", allow_pickle=False)
    theirs_a = np.load(Path(theirs_dir) / "per_atom_forces.npz", allow_pickle=False)

    pos = {s: i for i, s in enumerate(theirs_s["sid"].astype(str))}
    take = np.array([pos[s] for s in ours_s["sid"].astype(str)])
    o_off, t_off = ours_a["struct_offset"], theirs_a["struct_offset"]
    o_atoms = _ranges(o_off[:-1], o_off[1:])
    t_atoms = _ranges(t_off[take], t_off[take + 1])
    if len(o_atoms) != len(t_atoms):
        raise ValueError(f"{len(o_atoms)} atoms here against {len(t_atoms)} there")
    z_match = bool(np.array_equal(ours_a["Z"][o_atoms], theirs_a["Z"][t_atoms]))
    d_ref = np.abs(ours_a["ref_norm"][o_atoms] - theirs_a["ref_norm"][t_atoms])
    d_err = np.abs(ours_a["err_norm"][o_atoms] - theirs_a["err_norm"][t_atoms])
    rel = d_err / np.maximum(theirs_a["err_norm"][t_atoms], 1e-6)
    return {"n_structures": int(len(take)), "n_atoms": int(len(o_atoms)),
            "atom_order_matches": z_match,
            "max_abs_d_ref_norm_eV_per_A": float(d_ref.max()),
            "max_abs_d_err_norm_eV_per_A": float(d_err.max()),
            "median_rel_d_err_norm": float(np.median(rel)),
            "q999_rel_d_err_norm": float(np.quantile(rel, 0.999))}


def _restress(stress: np.ndarray, frm: tuple[float, str], to: tuple[float, str]) -> np.ndarray:
    """Re-read a stored Voigt-6 stress under a different sign / component order."""
    inv = np.empty(6, dtype=np.int64)
    inv[np.array(STRESS_ORDERS[frm[1]])] = np.arange(6)
    raw = stress / (frm[0] * KBAR_TO_EV_PER_A3)
    return to[0] * KBAR_TO_EV_PER_A3 * raw[:, inv][:, list(STRESS_ORDERS[to[1]])]


def calibrate_stress(model, frames: Frames, batches: list[np.ndarray], device, dtype) -> dict:
    """Which sign and component order of the release's stress a trained model agrees with.

    The release documents neither, so it is measured: the wrong sign or the wrong
    off-diagonal order leaves an error far larger than the model's own.
    """
    idx = np.concatenate(batches)
    pred = []
    for b in batches:
        batch = collate(frames, b).to(device=device, dtype=dtype)
        out = model(batch, compute_force=True, compute_stress=True, training=False)
        pred.append(out["stress"].detach().double()[:, _VOIGT_ROWS, _VOIGT_COLS].cpu().numpy())
    pred = np.concatenate(pred)
    stored = (float(frames.attrs.get("stress_sign", -1.0)), str(frames.attrs.get("stress_order", "vasp")))
    out = {}
    for sign in (stored[0], -stored[0]):
        for order in sorted(STRESS_ORDERS):
            ref = _restress(frames.stress[idx], stored, (sign, order))
            name = f"sign{'+' if sign > 0 else '-'}_{order}"
            out[name] = 1e3 * float(np.abs(pred - ref).mean())
    best = min(out, key=out.get)
    return {"stress_mae_meV_per_A3": out, "best": best, "stored": f"{stored[0]:+g}_{stored[1]}",
            "n_structures": int(len(idx))}


# -------------------------------------------------------------------------- CLI
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames", default=None, help="MatPES Frames HDF5 with neighbour lists")
    p.add_argument("--model", default=None, help="checkpoint (.pt) or mace-torch .model")
    p.add_argument("--out", default=None, help="output directory for this model's npz files")
    p.add_argument("--tables", nargs="+", default=None,
                   help="directories of per_structure.npz to aggregate instead of scoring")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--domain-from", default=None,
                   help="label whose per_structure.npz supplies the domain column")
    p.add_argument("--domain-rule", default="paper", choices=["paper", "stored"],
                   help="paper: relabel by dimensionality as the published tables do")
    p.add_argument("--dump-sids", default=None, metavar="PER_STRUCTURE_NPZ",
                   help="write the sid column of a per-structure file, one per line, to --out")
    p.add_argument("--compare", nargs=2, default=None, metavar=("OURS", "THEIRS"),
                   help="atom-by-atom agreement between two output directories")
    p.add_argument("--calibrate-stress", action="store_true",
                   help="measure the release's stress sign and component order")
    p.add_argument("--n", type=int, default=None, help="score a fixed random subset of this size")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-edges", type=int, default=60000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])


def run(args: argparse.Namespace) -> None:
    if args.dump_sids:
        sid = np.load(args.dump_sids, allow_pickle=False)["sid"].astype(str)
        Path(args.out).write_text("\n".join(sid) + "\n")
        print(f"wrote {len(sid):,} sids -> {args.out}", flush=True)
        return
    if args.compare:
        print(json.dumps(compare_per_atom(*args.compare), indent=2))
        return
    if args.tables:
        labels = args.labels or [Path(d).name for d in args.tables]
        runs, domain = align(_load_runs(args.tables, labels), args.domain_from, args.domain_rule)
        write_tables(tables(runs, domain), args.out or ".")
        return

    from ..eval.run_eval import load_model

    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    frames = Frames.load(args.frames)
    idx = np.arange(len(frames))
    if args.n is not None and args.n < len(frames):
        idx = np.sort(np.random.default_rng(0).choice(len(frames), args.n, replace=False))
    batches = eval_shards(idx, frames.nedges, 1, 0, args.batch_size, args.max_edges)
    model = load_model(args.model, device, dtype, args.backend)

    if args.calibrate_stress:
        print(json.dumps(calibrate_stress(model, frames, batches, device, dtype), indent=2))
        return

    per_structure, per_atom = score(model, frames, batches, device, dtype)
    if args.n is not None:  # keep only what was scored
        per_structure = {k: (v[idx] if getattr(v, "shape", ()) == (len(frames),) else v)
                         for k, v in per_structure.items()}
        atoms = _ranges(frames.atom_ptr[idx], frames.atom_ptr[idx + 1])
        per_atom = {**{k: per_atom[k][atoms] for k in ("d_force", "err_norm", "ref_norm", "Z")},
                    "frame": np.repeat(np.arange(len(idx), dtype=np.int32), frames.natoms[idx]),
                    "struct_offset": np.concatenate([[0], np.cumsum(frames.natoms[idx])])}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_structure.npz", **per_structure)
    np.savez_compressed(out / "per_atom_forces.npz", **per_atom)
    w = per_structure["natoms"].astype(float)
    print(json.dumps({"model": args.model, "n_structures": len(per_structure["sid"]),
                      "pooled_force_mae_meV_per_A":
                          1e3 * wmean(worst_fill(per_structure["f_mae_comp"]), w),
                      "pooled_energy_mae_meV_per_atom":
                          1e3 * wmean(worst_fill(per_structure["abs_d_energy_per_atom"]),
                                      np.ones_like(w)),
                      "non_finite_structures": int((~per_structure["f_finite"]).sum()),
                      "out": str(out)}, indent=2), flush=True)
