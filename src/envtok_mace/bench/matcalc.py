"""MatCalc-Bench for our model, with the paper's settings.

Four benchmarks from matcalc 0.5.1, run through our ASE calculator in float64:

  Equilibrium  WBM structures: relax, formation energy against tabulated MatPES-PBE
               elemental references ("all tabulated", as in the paper), and the
               structure fingerprint distance to the DFT relaxed structure
  Elasticity   MP binaries: bulk and shear moduli (VRH), relaxed-ion, 6 strains per axis
  Phonon       Alexandria binaries: heat capacity C_V
  Softening    WBM high-energy states: slope of predicted vs DFT forces

Structures with elements outside the model's table are dropped before sharding,
identically on every rank. ``--run`` shards the structures over ranks (RANK /
WORLD_SIZE from the launcher) and over ``--workers`` processes per GPU, since one
small relaxation leaves an A100 mostly idle. ``--merge`` joins the shards and
``--summary`` scores any set of run directories the way the paper's summary does,
so the authors' published CSVs and ours go through the same code.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from ..data.constants import Z_TABLE

ALLOWED_Z = frozenset(Z_TABLE)
BENCHES = ("Equilibrium", "Elasticity", "Phonon", "Softening")
ID_COLS = ("material_id", "mp_id", "wbm_id")

# quantity: (benchmark, prediction prefix, kind, unit)
QUANTITIES = {
    "eform": ("Equilibrium", "Eform_", "abs_error", "eV/atom"),
    "fpdist": ("Equilibrium", "d_", "value", "arb. units"),
    "softening": ("Softening", "softening_scale_", "deficit", "%"),
    "kvrh": ("Elasticity", "K_vrh_", "signed_pct", "%"),
    "gvrh": ("Elasticity", "G_vrh_", "signed_pct", "%"),
    "cv": ("Phonon", "CV_", "signed_pct", "%"),
    "kvrh_abs": ("Elasticity", "K_vrh_", "abs_pct", "%"),
    "gvrh_abs": ("Elasticity", "G_vrh_", "abs_pct", "%"),
    "cv_abs": ("Phonon", "CV_", "abs_pct", "%"),
}
STATS = ("median", "mean", "p5", "p25", "p75", "p95")


# ------------------------------------------------------------------ benchmarks
def _equilibrium_class():
    from matcalc import EnergeticsCalc
    from matcalc._stability import ELEMENTAL_REFS_DIR
    from matcalc.benchmark import EquilibriumBenchmark
    from monty.serialization import loadfn

    class TabulatedRefsEquilibrium(EquilibriumBenchmark):
        """Formation energies against tabulated MatPES-PBE elemental energies.

        Stock matcalc relaxes every elemental reference with the model, which
        fails for the molecular solids (O2, N2, F2, ...) an OMat24 model never
        saw. Reading every reference from the DFT table instead is the paper's
        mode. ``run`` also computes the fingerprint distance so that a failed
        relaxation (stored as NaN) is excluded rather than crashing the shard.
        """

        def __init__(self, *args, optimizer: str = "FIRE", **kwargs):
            super().__init__(*args, **kwargs)
            self.optimizer = optimizer
            self.table = loadfn(ELEMENTAL_REFS_DIR / "MatPES-PBE-Element-Refs.json.gz")

        def _refs(self) -> dict:
            needed = {el.symbol for s in self.structures for el in s.composition.elements}
            refs = {}
            for el in sorted(needed):
                epa = self.table[el]["energy_per_atom"]
                refs[el] = {"energy_per_atom": min(epa) if isinstance(epa, list) else epa}
            return refs

        def get_prop_calc(self, calculator, **kwargs):
            return EnergeticsCalc(
                calculator,
                elemental_refs=self._refs(),
                use_gs_reference=True,
                fmax=0.05,
                perturb_distance=0.1,
                optimizer=self.optimizer,
                **kwargs,
            )

        def run(self, calculator, model_name, **kwargs):
            from matminer.featurizers.site import CrystalNNFingerprint
            from matminer.featurizers.structure import SiteStatsFingerprint
            from pymatgen.core import Structure

            df = super(EquilibriumBenchmark, self).run(calculator, model_name, **kwargs)
            ssf = SiteStatsFingerprint(
                CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
                stats=("mean", "std_dev", "minimum", "maximum"),
            )

            def dist(a, b):
                if not isinstance(a, Structure) or not isinstance(b, Structure):
                    return np.nan
                try:
                    return float(np.linalg.norm(np.array(ssf.featurize(a)) - np.array(ssf.featurize(b))))
                except Exception:
                    return np.nan

            df[f"d_{model_name}"] = [
                dist(m, d) for m, d in zip(df[f"structure_{model_name}"], df["structure_DFT"])
            ]
            return df

    return TabulatedRefsEquilibrium


def build_benchmarks(names=BENCHES, limit: int | None = None, optimizer: str = "FIRE") -> dict:
    from matcalc.benchmark import ElasticityBenchmark, PhononBenchmark, SofteningBenchmark

    kw = {"n_samples": limit} if limit else {}
    make = {
        "Equilibrium": lambda: _equilibrium_class()(optimizer=optimizer, **kw),
        # matcalc's default strain envelopes (+-1 % normal, +-6 % shear) sampled at 6
        # points per axis instead of 4, and ions relaxed in every strained cell
        # (relaxed-ion constants, like the Materials Project reference).
        "Elasticity": lambda: ElasticityBenchmark(
            norm_strains=(-0.01, -0.00666667, -0.00333333, 0.00333333, 0.00666667, 0.01),
            shear_strains=(-0.06, -0.04, -0.02, 0.02, 0.04, 0.06),
            fmax=0.05,
            relax_calc_kwargs={"optimizer": optimizer},
            relax_deformed_structures=True,
            **kw,
        ),
        "Phonon": lambda: PhononBenchmark(optimizer=optimizer, **kw),
        "Softening": lambda: SofteningBenchmark(**kw),
    }
    return {n: make[n]() for n in names}


def filter_allowed(bench) -> int:
    """Drop structures with elements outside the model's table; return how many."""
    if not hasattr(bench, "structures"):
        return 0  # Softening is indexed by id only; filtered by formula at merge time
    keep = [
        i for i, s in enumerate(bench.structures) if all(el.Z in ALLOWED_Z for el in s.composition.elements)
    ]
    dropped = len(bench.structures) - len(keep)
    bench.structures = [bench.structures[i] for i in keep]
    gt = bench.ground_truth
    bench.ground_truth = (
        gt.iloc[keep].reset_index(drop=True) if hasattr(gt, "iloc") else [gt[i] for i in keep]
    )
    return dropped


def shard(bench, index: int, count: int) -> int:
    if hasattr(bench, "structures"):
        bench.structures = bench.structures[index::count]
        bench.ground_truth = bench.ground_truth[index::count]
        return len(bench.structures)
    bench.material_ids = bench.material_ids[index::count]
    return len(bench.material_ids)


# -------------------------------------------------------------------- running
def _log_failures(calc, path: Path) -> None:
    """matcalc turns any exception inside a calculation into NaN and keeps no
    trace of it. Record each calculator exception, so a NaN has a stated cause.
    CUDA running out of memory is retried once after emptying the allocator's
    cache: supercells of ever new shapes fragment it, and the retry usually fits."""
    import torch

    inner = calc.calculate

    def note(exc, retried=False):
        with path.open("a") as fh:
            first = str(exc).splitlines()[0][:200] if str(exc) else ""
            fh.write(
                f"{time.strftime('%H:%M:%S')} {type(exc).__name__}"
                f"{' (after retry)' if retried else ''}: {first}\n"
            )

    def calculate(*args, **kwargs):
        try:
            return inner(*args, **kwargs)
        except torch.OutOfMemoryError as exc:
            note(exc)
            torch.cuda.empty_cache()
            try:
                return inner(*args, **kwargs)
            except Exception as again:
                note(again, retried=True)
                raise
        except Exception as exc:
            note(exc)
            raise

    calc.calculate = calculate


def _run_shard(
    model: str,
    label: str,
    out: str,
    names: tuple,
    index: int,
    count: int,
    device: str,
    dtype: str,
    limit: int | None,
    threads: int,
    backend: str | None = None,
) -> None:
    import torch

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    out_dir = Path(out)
    calc = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend)
    tag = f"shard{index}of{count}"
    _log_failures(calc, out_dir / f".calc_errors_{label}_{tag}.log")
    for name, bench in build_benchmarks(names, limit).items():
        csv = out_dir / f"{name}_{label}_{tag}.csv"
        if csv.exists():
            continue
        dropped = filter_allowed(bench)
        n = shard(bench, index, count)
        if n == 0:
            continue
        t0 = time.time()
        ckpt = out_dir / f".ckpt_{name}_{tag}.json"
        if name == "Softening":
            df = bench.run(calc, label, checkpoint_file=str(ckpt), checkpoint_freq=10)
        else:
            df = bench.run(
                calc,
                label,
                n_jobs=1,
                checkpoint_file=str(ckpt),
                checkpoint_freq=10,
                delete_checkpoint_on_finish=True,
            )
        df = df[[c for c in df.columns if not c.startswith("structure_")]]
        df.to_csv(csv, index=False)
        ckpt.unlink(missing_ok=True)
        print(
            f"[{tag}] {name}: {n} structures in {(time.time() - t0) / 60:.1f} min "
            f"({dropped} dropped for elements) -> {csv.name}",
            flush=True,
        )


def run_benchmarks(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = args.device
    if device == "cuda":
        import torch

        device = f"cuda:{local % max(torch.cuda.device_count(), 1)}"
    k = args.workers
    count = world * k
    Path(args.out).mkdir(parents=True, exist_ok=True)
    names = tuple(args.only or BENCHES)
    jobs = [
        (
            args.model,
            args.label,
            args.out,
            names,
            rank * k + i,
            count,
            device,
            args.dtype,
            args.limit,
            max(1, args.threads // k),
            args.backend,
        )
        for i in range(k)
    ]
    if k == 1:
        _run_shard(*jobs[0])
        return
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_run_shard, args=j) for j in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise SystemExit(f"{len(bad)} of {k} workers failed (exit codes {bad})")


# --------------------------------------------------------------- merge / score
def _disallowed(formula) -> bool:
    from pymatgen.core import Composition

    try:
        return any(el.Z not in ALLOWED_Z for el in Composition(str(formula)).elements)
    except Exception:
        return False


def merge(out: str | Path, label: str) -> None:
    import pandas as pd

    out = Path(out)
    for name in BENCHES:
        parts = sorted(out.glob(f"{name}_{label}_shard*of*.csv"))
        if not parts:
            continue
        counts = {p.name.rsplit("of", 1)[1] for p in parts}
        total = int(next(iter(counts)).split(".")[0])
        if len(counts) != 1 or len(parts) != total:
            print(f"{name}: {len(parts)} shard files for {counts} shards -- incomplete, not merged")
            continue
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        idc = next(c for c in ID_COLS if c in df.columns)
        if df[idc].duplicated().any():
            raise ValueError(f"{name}: repeated ids after merging")
        if "formula" in df.columns:
            df = df[~df["formula"].apply(_disallowed)]
        path = out / f"{name}_{label}_FULL.csv"
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} rows -> {path.name}")


def values(df, key: str, unphysical: str = "drop") -> dict:
    """Valid values of one quantity, as in the paper's summary: failed rows dropped,
    and with ``drop`` non-positive moduli or heat capacities too."""
    import pandas as pd

    _, prefix, kind, _ = QUANTITIES[key]
    cands = [c for c in df.columns if c.startswith(prefix) and not c.endswith("_DFT")]
    if len(cands) != 1:
        raise ValueError(f"expected one prediction column {prefix}<model>, got {cands}")
    pred = pd.to_numeric(df[cands[0]], errors="coerce").to_numpy(float)
    ok = np.isfinite(pred)
    n_unphysical = 0
    ref = None
    if kind != "value" and kind != "deficit":
        ref = pd.to_numeric(df[f"{prefix}DFT"], errors="coerce").to_numpy(float)
        ok &= np.isfinite(ref)
    if kind in ("signed_pct", "abs_pct"):
        ok &= ref > 0
        bad = ok & (pred <= 0)
        n_unphysical = int(bad.sum())
        if unphysical == "drop":
            ok &= ~bad
    p = pred[ok]
    if kind == "abs_error":
        v = np.abs(p - ref[ok])
    elif kind == "value":
        v = p
    elif kind == "deficit":
        v = 100.0 * (1.0 - p)
    else:
        v = 100.0 * (p - ref[ok]) / ref[ok]
        if kind == "abs_pct":
            v = np.abs(v)
    return {
        "values": v,
        "n_rows": int(len(df)),
        "n_failed": int((~np.isfinite(pred)).sum()),
        "n_unphysical": n_unphysical,
    }


def stats(v: np.ndarray) -> dict:
    if v.size == 0:
        return {s: float("nan") for s in STATS}
    q = np.percentile(v, [5, 25, 75, 95])
    return {
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "p5": q[0],
        "p25": q[1],
        "p75": q[2],
        "p95": q[3],
    }


def summary(dirs: list[str], labels: list[str], unphysical: str = "drop") -> dict:
    """Per model and quantity: n, failed, unphysical and the distribution of the error.
    Labels starting with ``rdm`` are also pooled into ``random``."""
    import pandas as pd

    per: dict = {}
    for label, d in zip(labels, dirs):
        frames = {}
        for name in BENCHES:
            hits = sorted(Path(d).glob(f"{name}_*_FULL.csv"))
            if len(hits) == 1:
                frames[name] = pd.read_csv(hits[0])
        for key, (bench, prefix, *_rest) in QUANTITIES.items():
            df = frames.get(bench)
            if df is not None and any(c.startswith(prefix) and not c.endswith("_DFT") for c in df.columns):
                per.setdefault(key, {})[label] = values(df, key, unphysical)
    out = {"unphysical": unphysical, "quantities": {}}
    for key, models in per.items():
        rows = {}
        seeds = [L for L in models if L.lower().startswith("rdm")]
        groups = {L: [L] for L in models}
        if seeds:
            groups["random"] = seeds
        for g, members in groups.items():
            v = np.concatenate([models[m]["values"] for m in members])
            rows[g] = {
                "n": int(v.size),
                "n_failed": sum(models[m]["n_failed"] for m in members),
                "n_unphysical": sum(models[m]["n_unphysical"] for m in members),
                **stats(v),
            }
        out["quantities"][key] = {"unit": QUANTITIES[key][3], "models": rows}
    return out


# ------------------------------------------------------------------------ CLI
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="checkpoint (.pt) or mace-torch .model")
    p.add_argument("--label", default="ours", help="model name in the CSV columns and file names")
    p.add_argument("--out", default="matcalc_results")
    p.add_argument("--only", nargs="*", default=None, choices=list(BENCHES))
    p.add_argument("--limit", type=int, default=None, help="random subset per benchmark (timing)")
    p.add_argument("--workers", type=int, default=1, help="processes per GPU")
    p.add_argument("--threads", type=int, default=16, help="CPU threads per GPU, split over workers")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument(
        "--backend",
        default=None,
        choices=[None, "e3nn", "cueq"],
        help="default: the backend the model was trained with",
    )
    p.add_argument("--merge", action="store_true", help="join this label's shard CSVs")
    p.add_argument(
        "--summary",
        nargs="+",
        default=None,
        metavar="DIR",
        help="score these run directories (with --labels) instead of running",
    )
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--unphysical", default="drop", choices=["drop", "keep"])


def run(args: argparse.Namespace) -> None:
    if args.summary:
        labels = args.labels or [Path(d).name for d in args.summary]
        res = summary(args.summary, labels, args.unphysical)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "matcalc_summary.json").write_text(json.dumps(res, indent=1))
        for key, q in res["quantities"].items():
            scale, unit = (1e3, "meV/atom") if key == "eform" else (1.0, q["unit"])
            print(f"== {key} [{unit}]  (median / mean / p95, n, failed+unphysical)")
            for g, r in q["models"].items():
                print(
                    f"  {g:16s} {scale * r['median']:9.3f} {scale * r['mean']:9.3f} "
                    f"{scale * r['p95']:9.3f}   n={r['n']:5d}  excluded={r['n_failed'] + r['n_unphysical']}"
                )
        return
    if args.merge:
        merge(args.out, args.label)
        return
    if not args.model:
        raise SystemExit("--model is required to run the benchmarks")
    run_benchmarks(args)
