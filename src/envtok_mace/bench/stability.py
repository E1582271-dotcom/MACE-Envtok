"""MLIP-Arena MD stability: NPT compression of the RM24 structures, the paper's protocol.

Each run heats 300 -> 3000 K while compressing 0 -> 500 GPa over 10 ps (Nose-Hoover,
2 fs, or 0.5 fs with hydrogen), using mlip-arena's own MD task so the dynamics are
the published ones. The structures are the first 80 RM24 random mixtures whose
elements are all in the model's table.

The arena caps every run at three attempts of 600 s wall time each, resuming
from the trajectory, so the completion score depends on the speed of a step as
well as on stability. One run per GPU at a time, as in the paper; the per-step
speed is recorded so that it can be compared with the authors' runs.

Outcomes follow the paper's classification, in order of precedence: survived
(reached >= 0.999 of the run), oom, diverged (a singular NPT cell, or with no
failure marker a temperature above 1e5 K or non-finite), other failure,
truncated (still running when the wall-clock budget ran out), missing.
``--summary`` applies it to any set of result directories, the authors' included.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..data.constants import Z_TABLE

ALLOWED_Z = frozenset(Z_TABLE)
SURVIVED = 0.999       # normalized final step of a finished run (~0.9996)
BUDGET_S = 3 * 600.0   # wall-clock budget per run: three attempts of 600 s
T_BLOWUP = 1e5         # K; the run targets <= 3000 K
OUTCOMES = ("survived", "diverged", "oom", "other", "truncated", "missing")
TOTAL_PS, T_START, T_END, P_END_GPA = 10.0, 300.0, 3000.0, 500.0
_IDX_RE = re.compile(r"_(\d+)_npt$")


# ----------------------------------------------------------------------- inputs
def load_rm24(db: str | None = None) -> list:
    """All RM24 structures in database order (the HF file is staged on a login node)."""
    from ase.db import connect

    if db is None or not Path(db).exists():
        from huggingface_hub import hf_hub_download

        db = hf_hub_download(repo_id="atomind/mlip-arena", repo_type="dataset",
                             subfolder="stability", filename="random-mixture.db")
    with connect(db) as conn:
        return [row.toatoms() for row in conn.select()]


def npt_worklist(structures: list, n_npt: int = 80) -> list:
    """(index, atoms) of the first ``n_npt`` structures with supported elements only."""
    ok = [a for a in structures if ALLOWED_Z.issuperset(a.get_atomic_numbers().tolist())]
    return list(enumerate(ok[:n_npt]))


def traj_path(out: Path, label: str, idx: int, atoms) -> Path:
    mid = atoms.info.get("material_id", "random")
    return out / f"{label}_{mid}_{atoms.get_chemical_formula()}_{idx}_npt.traj"


# ----------------------------------------------------------------------- running
class _Timeout(Exception):
    pass


@contextmanager
def wall_timeout(seconds: float):
    """SIGALRM guard: interrupts the MD loop between steps after ``seconds``."""
    def handler(signum, frame):
        raise _Timeout(f"run timed out after {seconds:g} s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_one(idx: int, atoms, calc, out: Path, label: str, timeout: float = 600.0,
            attempts: int = 3, total_fs: float = 1e4) -> str:
    """One NPT compression run, with the arena's retries; markers record the outcome."""
    from ase import units
    from mlip_arena.tasks import MD

    path = traj_path(out, label, idx, atoms)
    done = path.with_suffix(".done")
    if done.exists():
        return "skip"
    kw = dict(atoms=atoms, calculator=calc, ensemble="npt", dynamics="nose-hoover",
              time_step=None, total_time=total_fs, temperature=[T_START, T_END],
              pressure=[0, P_END_GPA * units.GPa], traj_file=str(path), traj_interval=10,
              restart=True,
              dynamics_kwargs=dict(ttime=25.0 * units.fs,
                                   pfactor=((75 * units.fs) ** 2) * 1e2 * units.GPa))
    last = None
    finished = False
    for _ in range(attempts):
        try:
            with wall_timeout(timeout):
                MD.fn(**kw)
            finished = True
            break
        except _Timeout:
            last = "timeout"
        except Exception:
            tb = traceback.format_exc()
            (out / f".failed_{path.stem}.txt").write_text(tb)
            if "CUDA error" in tb:
                # A device fault (e.g. an illegal memory access) leaves this process's
                # CUDA context unusable, so every later run here would fail at once.
                # It says nothing about the model: leave the run unfinished for a rerun
                # and stop this rank.
                raise SystemExit(f"CUDA device error in {path.name}; rank stopped, rerun needed")
            last = "oom" if "out of memory" in tb.lower() else "crash"
        finally:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    done.write_text("done\n")
    if finished:
        return "finished"
    if last == "oom":
        (out / f".oom_{path.stem}.txt").write_text("cuda-oom\n")
    return last or "crash"


def run_benchmark(args: argparse.Namespace) -> None:
    from ..eval.ase_calc import EnvTokCalculator

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = args.device
    if device == "cuda":
        import torch

        device = f"cuda:{local % max(torch.cuda.device_count(), 1)}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = npt_worklist(load_rm24(args.db), args.n_npt)
    if args.limit:
        work = work[:args.limit]
    calc = EnvTokCalculator(args.model, device=device, dtype=args.dtype, backend=args.backend)
    for idx, atoms in work[rank::world]:
        t0 = time.time()
        status = run_one(idx, atoms, calc, out, args.label, args.timeout, args.attempts,
                         args.total_fs)
        print(f"[rank {rank}] npt {idx} {atoms.get_chemical_formula()} ({len(atoms)} atoms): "
              f"{status} in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ aggregation
def trajectory_frames(path: Path):
    """Per-frame rows of one trajectory, computed like the arena's runtime stats."""
    import pandas as pd
    from ase.io import read

    try:
        traj = read(path, index=":")
    except Exception as exc:  # e.g. an empty file from a run that never took a step
        print(f"skipping {path.name}: {exc}")
        return None
    steps, temps, restarts, times = [], [], [], []
    for atoms in traj:
        try:
            e = atoms.get_potential_energy()
        except Exception:
            continue
        if not np.isfinite(e) or not {"restart", "datetime", "step"} <= atoms.info.keys():
            continue
        steps.append(atoms.info["step"])
        temps.append(atoms.get_temperature())
        restarts.append(atoms.info["restart"])
        times.append(atoms.info["datetime"])
    target = next((a.info["target_steps"] for a in traj if "target_steps" in a.info), 0)
    if not steps or not target:
        return None
    steps, restarts, times = np.array(steps), np.array(restarts), np.array(times)
    seconds = n = 0
    for block in np.unique(restarts):
        m = restarts == block
        seconds += (times[m][-1] - times[m][0]).total_seconds()
        n += steps[m][-1] - steps[m][0]
    natoms = len(traj[0])
    return pd.DataFrame({
        "source": path.stem, "formula": traj[0].get_chemical_formula(), "natoms": natoms,
        "normalized_final_step": steps[-1] / target, "temperatures": temps,
        "timestep": steps, "target_steps": target,
        "seconds_per_step": seconds / n if n else np.inf,
        "seconds_per_step_per_atom": seconds / n / natoms if n else np.inf,
    })


def aggregate(out: str | Path, label: str) -> Path:
    import pandas as pd

    out = Path(out)
    parts = [trajectory_frames(p) for p in sorted(out.glob(f"{label}_*_npt.traj"))]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    path = out / f"{label}-compression.parquet"
    df.to_parquet(path)
    print(f"{df['source'].nunique()} trajectories -> {path.name}")
    return path


# ---------------------------------------------------------------- classification
def marker_mode(text: str) -> str:
    low = text.lower()
    if "out of memory" in low or "outofmemoryerror" in low or "cuda-oom" in low:
        return "oom"
    if "singular matrix" in low:
        return "singular_matrix"
    return "other"


def classify(run_dir: str | Path, n_npt: int = 80):
    """One row per protocol NPT run (index 0..n_npt-1) with its outcome."""
    import pandas as pd

    run_dir = Path(run_dir)
    hits = sorted(run_dir.glob("*-compression.parquet"))
    if len(hits) != 1:
        raise ValueError(f"expected one *-compression.parquet in {run_dir}, found {hits}")
    frames = pd.read_parquet(hits[0], columns=["source", "natoms", "normalized_final_step",
                                               "temperatures", "seconds_per_step",
                                               "target_steps"])
    temps = pd.to_numeric(frames["temperatures"], errors="coerce")
    frames = frames.assign(bad_t=~np.isfinite(temps) | (temps.abs() > T_BLOWUP))
    g = frames.groupby("source", sort=True)
    per = pd.DataFrame({"natoms": g["natoms"].first(),
                        "nfs": pd.to_numeric(g["normalized_final_step"].first(), errors="coerce"),
                        "temp_blowup": g["bad_t"].any(),
                        "seconds_per_step": g["seconds_per_step"].first(),
                        "target_steps": g["target_steps"].first()}).reset_index()
    per["idx"] = per["source"].str.extract(_IDX_RE)[0].astype(int)
    per = per[per["idx"] < n_npt]

    modes = {}
    for prefix in (".failed_", ".oom_"):  # .oom_ last: it names the final attempt
        for p in sorted(run_dir.glob(f"{prefix}*_npt.txt")):
            modes[p.name[len(prefix):-len(".txt")]] = marker_mode(p.read_text(errors="replace"))
    per["marker"] = per["source"].map(modes)

    out = np.full(len(per), "truncated", dtype=object)
    out[(per["marker"] == "other").to_numpy()] = "other"
    out[((per["marker"] == "singular_matrix")
         | (per["marker"].isna() & per["temp_blowup"])).to_numpy()] = "diverged"
    out[(per["marker"] == "oom").to_numpy()] = "oom"
    out[(per["nfs"] >= SURVIVED).to_numpy()] = "survived"
    out[per["nfs"].isna().to_numpy()] = "missing"
    per["outcome"] = out
    absent = sorted(set(range(n_npt)) - set(per["idx"]))
    if absent:
        per = pd.concat([per, pd.DataFrame({"idx": absent, "outcome": "missing",
                                            "temp_blowup": False})], ignore_index=True)
    per["completion"] = per["nfs"].fillna(0.0).clip(lower=0.0)
    return per.sort_values("idx").reset_index(drop=True)


def speed_matched(per, ref):
    """``per`` as it would have scored at the step speed of the reference runs ``ref``.

    The protocol's 1,800 s wall-clock budget truncates slow runs, so a faster
    implementation finishes more of them. For each structure, cut the run where
    the reference setup would have run out of time; an outcome later than that
    point becomes 'truncated'.
    """
    reach = (BUDGET_S / (ref["seconds_per_step"] * ref["target_steps"])).clip(upper=1.0)
    reach = per["idx"].map(dict(zip(ref["idx"], reach)))
    out = per.copy()
    cut = reach.notna() & (reach < SURVIVED) & (out["nfs"] > reach)
    out.loc[cut, "outcome"] = "truncated"
    out.loc[cut, "nfs"] = reach[cut]
    out["completion"] = out["nfs"].fillna(0.0).clip(lower=0.0)
    return out


def rates(per) -> dict:
    n = len(per)
    r = {"n_runs": n}
    for o in OUTCOMES:
        r[f"n_{o}"] = int((per["outcome"] == o).sum())
        r[f"{o}_pct"] = 100.0 * r[f"n_{o}"] / n
    r["phys_div_incl_oom_pct"] = 100.0 * (r["n_diverged"] + r["n_oom"]) / n
    comp = per["completion"].to_numpy(float)
    r["mean_completion_pct"] = 100.0 * comp.mean()
    ended = per["outcome"].isin(["diverged", "oom", "other", "truncated"])
    for tag, mask in (("first_end", ended), ("first_div", per["outcome"] == "diverged")):
        frac = float(per.loc[mask, "nfs"].min()) if mask.any() else np.nan
        r[f"{tag}_ps"] = frac * TOTAL_PS
        r[f"{tag}_K"] = T_START + frac * (T_END - T_START)
        r[f"{tag}_GPa"] = frac * P_END_GPA
    return r


def summary(dirs: list[str], labels: list[str], n_npt: int = 80,
            speed_ref: str | None = None) -> dict:
    """Rates per model; labels starting with ``rdm`` are also pooled into ``random``.
    With ``speed_ref`` (a label), every other model is also scored at that model's
    step speed, as ``<label>@speed_of_<ref>``."""
    import pandas as pd

    per = {L: classify(d, n_npt) for L, d in zip(labels, dirs)}
    out = {L: rates(p) for L, p in per.items()}
    if speed_ref:
        for L in labels:
            if L != speed_ref:
                out[f"{L}@speed_of_{speed_ref}"] = rates(speed_matched(per[L], per[speed_ref]))
    seeds = [L for L in labels if L.lower().startswith("rdm")]
    if seeds:
        out["random"] = rates(pd.concat([per[s] for s in seeds], ignore_index=True))
    return out


# -------------------------------------------------------------------------- CLI
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None)
    p.add_argument("--label", default="ours")
    p.add_argument("--out", default="stability_results")
    p.add_argument("--db", default=None, help="random-mixture.db (default: HF cache)")
    p.add_argument("--n-npt", type=int, default=80)
    p.add_argument("--limit", type=int, default=None, help="only the first N runs (timing)")
    p.add_argument("--timeout", type=float, default=600.0, help="wall seconds per attempt")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--total-fs", type=float, default=1e4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])
    p.add_argument("--aggregate", action="store_true", help="write this label's parquet")
    p.add_argument("--summary", nargs="+", default=None, metavar="DIR")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--speed-ref", default=None,
                   help="label whose step speed the other models are also scored at")


def run(args: argparse.Namespace) -> None:
    if args.summary:
        import json

        labels = args.labels or [Path(d).name for d in args.summary]
        res = summary(args.summary, labels, args.n_npt, args.speed_ref)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "stability_summary.json").write_text(json.dumps(res, indent=1))
        print(f"{'model':36s} {'complete%':>9s} {'survived':>8s} {'diverged':>8s} {'oom':>4s} "
              f"{'other':>5s} {'trunc':>5s} {'first end ps':>12s}")
        for L, r in res.items():
            print(f"{L:36s} {r['mean_completion_pct']:9.1f} {r['n_survived']:8d} "
                  f"{r['n_diverged']:8d} {r['n_oom']:4d} {r['n_other']:5d} "
                  f"{r['n_truncated']:5d} {r['first_end_ps']:12.2f}")
        return
    if args.aggregate:
        aggregate(args.out, args.label)
        return
    if not args.model:
        raise SystemExit("--model is required to run the benchmark")
    run_benchmark(args)
