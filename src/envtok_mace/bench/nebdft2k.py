"""LiTraj nebDFT2k: Li-ion migration barriers by nudged elastic band, the paper's protocol.

For each of the dataset's Li-vacancy hops, start from its BVSE-NEB band, relax
both end-points (FIRE, fmax 0.05 eV/A, up to 10,000 steps -- the paper's
symmetric variant of LiTraj's recipe), then the climbing-image band (k = 5 eV/A^2,
improved tangent, FIRE to fmax 0.05, up to 10,000 steps). The barrier is
max(E) - min(E) over the band (litraj.utils.get_barrier). A hop is scored only if
the largest force component on the band and end-points is below 0.1 eV/A,
following the authors' convergence check.

The DFT reference is plain PBE without Hubbard U, while OMat24, and so the
model, uses U for oxides and fluorides of Co, Cr, Fe, Mn, Mo, Ni, V and W. Hops
in those hosts (``has_specific_TM``) are reported separately; the U-free split
is the like-for-like comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from ..data.constants import Z_TABLE
from ._runner import Timeout, failure, placement, spawn, wall_timeout

ALLOWED_Z = frozenset(Z_TABLE)
DATASET = "nebDFT2k"
GATE = 0.1  # eV/A, largest force component over band and end-points


def _unsupported(chemsys: str) -> bool:
    from ase.data import atomic_numbers

    return any(atomic_numbers[e] not in ALLOWED_Z for e in str(chemsys).split("-"))


def prefetch(cache: str | Path) -> None:
    """Login node: download and unpack the dataset (65 MB) with litraj's own loader."""
    from litraj.data import download_dataset, load_index

    Path(cache).mkdir(parents=True, exist_ok=True)
    download_dataset(DATASET, str(cache), unzip=True)
    print(f"{len(load_index(DATASET, str(cache)))} hops indexed in {cache}")


def worklist(cache: str | Path) -> list[dict]:
    from litraj.data import load_index

    idx = load_index(DATASET, str(cache))
    idx = idx[~idx["chemsys"].map(_unsupported)]
    return idx.to_dict("records")


def relax_band(images, calcs, fmax: float = 0.05, steps: int = 10_000, k: float = 5.0) -> dict:
    """Relax end-points and the climbing-image band in place; return diagnostics."""
    from ase.mep import NEB
    from ase.optimize import FIRE

    for image, calc in zip(images, calcs):  # NEB needs one calculator object per image
        image.calc = calc
    quiet = dict(logfile=os.devnull)
    src = FIRE(images[0], **quiet)
    src.run(fmax=fmax, steps=steps)
    tgt = FIRE(images[-1], **quiet)
    tgt.run(fmax=fmax, steps=steps)
    neb = NEB(images, parallel=False, k=k, climb=True, method="improvedtangent")
    band = FIRE(neb, **quiet)
    band.run(fmax=fmax, steps=steps)
    worst = max(
        float(np.abs(neb.get_forces()).max()),
        float(np.abs(images[0].get_forces()).max()),
        float(np.abs(images[-1].get_forces()).max()),
    )
    return {
        "steps_source": src.get_number_of_steps(),
        "steps_target": tgt.get_number_of_steps(),
        "steps_band": band.get_number_of_steps(),
        "fmax_worst": worst,
        "converged": bool(worst < GATE),
    }


def run_hop(row: dict, calcs, cache: Path, out: Path, label: str, timeout: float, attempts: int = 2) -> str:
    from ase.io import read
    from litraj.metrics import PA_MAE
    from litraj.utils import get_barrier

    edge = row["edge_id"]
    path = out / f"{label}_{edge}.json"
    done = path.with_suffix(".done")
    if done.exists():
        return "skip"
    init = read(str(cache / DATASET / f"{edge}_init.xyz"), index=":")
    ref = read(str(cache / DATASET / f"{edge}_relaxed.xyz"), index=":")
    result, last = None, None
    for _ in range(attempts):  # a NEB cannot resume, so every attempt starts afresh
        try:
            with wall_timeout(timeout, "hop"):
                images = [im.copy() for im in init]
                diag = relax_band(images, calcs[: len(images)])
                result = {
                    **diag,
                    "em_pred": float(get_barrier(images)),
                    "pa_mae": float(PA_MAE(ref, images)) if len(ref) == len(images) else None,
                    "n_images": len(images),
                    "natoms": len(images[0]),
                }
            break
        except Timeout:
            last = "timeout"
        except Exception:
            tb = traceback.format_exc()
            (out / f".failed_{path.stem}.txt").write_text(tb)
            last = failure(tb, edge)
    done.write_text("done\n")
    if result is None:
        return last or "crash"
    result.update(
        edge_id=edge,
        chemsys=row["chemsys"],
        em_dft=float(row["em_dft"]),
        has_specific_TM=bool(row["has_specific_TM"]),
    )
    path.write_text(json.dumps(result))
    return "converged" if result["converged"] else "not_converged"


def run_benchmark(args: argparse.Namespace) -> None:
    rank, world, device = placement(args.device)
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    work = worklist(cache)
    if args.limit:
        work = work[: args.limit]
    # Shard only the hops that are left, so a rerun spreads over all workers.
    work = [r for r in work if not (out / f"{args.label}_{r['edge_id']}.done").exists()]
    k = args.workers
    shards = [(rank * k + i, world * k) for i in range(k)]
    if k == 1:
        _run_shard(args, device, out, cache, work, *shards[0])
        return
    # exit codes are not checked: a worker stopped by a CUDA fault leaves its hops for the rerun
    spawn(_run_shard, [(args, device, out, cache, work, i, n) for i, n in shards], check=False)


def _run_shard(args, device, out, cache, work, index, count) -> None:
    import torch

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(max(1, args.threads // args.workers))
    first = EnvTokCalculator(args.model, device=device, dtype=args.dtype, backend=args.backend)
    # images share the model weights; each needs its own calculator object for NEB
    calcs = [first] + [EnvTokCalculator(first.model, device=device, dtype=args.dtype) for _ in range(8)]
    for row in work[index::count]:
        t0 = time.time()
        status = run_hop(row, calcs, cache, out, args.label, args.timeout)
        print(f"[shard {index}/{count}] {row['edge_id']}: {status} in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------- scoring
def load_results(run_dir: str | Path, label: str | None = None):
    """One row per hop with a result: ours from the JSON files, or the authors' parquet."""
    import pandas as pd

    run_dir = Path(run_dir)
    hits = sorted(run_dir.glob("*-litraj-nebdft2k.parquet"))
    if hits:
        df = pd.read_parquet(hits[0]).drop(columns=["profile_pred"], errors="ignore")
    else:
        pattern = f"{label}_*.json" if label else "*.json"
        df = pd.DataFrame([json.loads(p.read_text()) for p in sorted(run_dir.glob(pattern))])
    oom = df["oom_terminated"].astype(bool) if "oom_terminated" in df else pd.Series(False, index=df.index)
    unsupported = (
        df["has_unsupported"].astype(bool) if "has_unsupported" in df else df["chemsys"].map(_unsupported)
    )
    df["kept"] = (
        df["converged"].astype(bool)
        & ~oom
        & ~unsupported
        & np.isfinite(df["em_pred"])
        & np.isfinite(df["em_dft"])
    )
    df["err"] = df["em_pred"] - df["em_dft"]
    return df


def metrics(err: np.ndarray) -> dict:
    if err.size == 0:
        return {"n_kept": 0}
    a = np.abs(err)
    return {
        "n_kept": int(err.size),
        "mae": float(a.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "med_abs": float(np.median(a)),
        "median_signed": float(np.median(err)),
        "mean_signed": float(err.mean()),
    }


def summary(dirs: list[str], labels: list[str]) -> dict:
    out = {}
    for label, d in zip(labels, dirs):
        df = load_results(d, label)
        res = {}
        for name, with_u in (("without_U", False), ("with_U", True)):
            sub = df[df["has_specific_TM"].astype(bool) == with_u]
            res[name] = {
                **metrics(sub.loc[sub["kept"], "err"].to_numpy(float)),
                "n_results": int(len(sub)),
                "n_not_converged": int((~sub["converged"].astype(bool)).sum()),
            }
        out[label] = res
    seeds = [L for L in labels if L.lower().startswith("rdm")]
    if len(seeds) > 1:
        for name in ("without_U", "with_U"):
            for key in ("mae", "rmse", "med_abs"):
                v = [out[s][name][key] for s in seeds]
                out.setdefault("random", {}).setdefault(name, {})[key] = float(np.mean(v))
                out["random"][name][f"{key}_sd"] = float(np.std(v, ddof=1))
    return out


def hops_table(dirs: list[str], labels: list[str]):
    """One row per hop: the DFT barrier and, per model, the predicted barrier and
    whether the hop is scored (converged band, supported elements)."""

    out = None
    for label, d in zip(labels, dirs):
        df = load_results(d, label)[["edge_id", "chemsys", "has_specific_TM", "em_dft", "em_pred", "kept"]]
        df = df.rename(columns={"em_pred": f"em_pred_{label}", "kept": f"kept_{label}"})
        if out is None:
            out = df
        else:
            out = out.merge(
                df.drop(columns=["chemsys", "has_specific_TM", "em_dft"]), on="edge_id", how="outer"
            )
    return out.sort_values("edge_id").reset_index(drop=True)


# -------------------------------------------------------------------------- CLI
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None)
    p.add_argument("--label", default="ours")
    p.add_argument("--out", default="nebdft2k_results")
    p.add_argument("--cache", default="litraj_cache", help="directory of the prefetched dataset")
    p.add_argument("--prefetch", action="store_true", help="download the dataset (login node)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=3600.0, help="wall seconds per attempt")
    p.add_argument("--workers", type=int, default=1, help="processes per GPU")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])
    p.add_argument("--summary", nargs="+", default=None, metavar="DIR")
    p.add_argument("--labels", nargs="+", default=None)


def run(args: argparse.Namespace) -> None:
    if args.prefetch:
        prefetch(args.cache)
        return
    if args.summary:
        labels = args.labels or [Path(d).name for d in args.summary]
        res = summary(args.summary, labels)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "nebdft2k_summary.json").write_text(json.dumps(res, indent=1))
        hops_table(args.summary, labels).to_csv(
            Path(args.out) / "nebdft2k_hops.csv", index=False, float_format="%.5f"
        )
        for split in ("without_U", "with_U"):
            print(f"== {split}   MAE / RMSE / median|err| (eV), n kept")
            for L, r in res.items():
                m = r[split]
                print(
                    f"  {L:16s} {m.get('mae', np.nan):.3f} {m.get('rmse', np.nan):.3f} "
                    f"{m.get('med_abs', np.nan):.3f}  n={m.get('n_kept', '-')}"
                )
        return
    if not args.model:
        raise SystemExit("--model is required to run the benchmark")
    run_benchmark(args)
