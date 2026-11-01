"""Training-step throughput of our model and of the reference mace-torch model
(with and without cuEquivariance kernels) on the same batches, one GPU.

A step is forward (energy, forces, stress), loss, backward and an Adam update.

With ``--mode infer``: the latency and memory of one ASE calculator call against system
size, on fcc copper, a dense metal, so the edges per atom are at the high end of real
structures. Each precision and backend runs in a process of its own, so the peak memory
of one does not carry over into the next. A size that runs out of memory, or whose first
call exceeds ``--max-seconds``, is the last one tried.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np
import torch

from ..data.batch import collate
from ..data.frames import Frames
from ..model.mace import MACE, MACEConfig
from ..train.loss import LossConfig, combine, loss_counts, loss_sums
from ._runner import spawn

INFER_REPEATS = (1, 2, 3, 4, 5, 7, 10, 13, 16, 18)  # fcc cubic cell: 4 to 23,328 atoms


def _make_model(impl: str, device: torch.device):
    torch.manual_seed(0)
    cfg = MACEConfig()
    if impl == "ours":
        return MACE(cfg).to(device), None
    if impl == "ours-cueq":
        return MACE(MACEConfig(backend="cueq")).to(device), None
    from ..model.mace_compat import build_reference_mace, mace_input_dict

    ref = build_reference_mace(cfg).to(device)
    if impl == "mace-cueq":
        from mace.cli.convert_e3nn_cueq import run as to_cueq

        ref = to_cueq(ref, device=device.type).to(device)
    return ref, mace_input_dict


def benchmark(
    frames: Frames, impl: str, batch_size: int, steps: int, warmup: int, device: torch.device, seed: int = 0
) -> dict:
    torch.set_default_dtype(torch.float32)
    model, to_mace = _make_model(impl, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    cfg = LossConfig()
    rng = np.random.default_rng(seed)
    batches = [np.sort(rng.choice(len(frames), batch_size, replace=False)) for _ in range(warmup + steps)]
    n_atoms = n_structs = 0
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t0 = None
    for i, idx in enumerate(batches):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        batch = collate(frames, idx).to(device=device, dtype=torch.float32)
        if to_mace is None:
            out = model(batch, compute_force=True, compute_stress=True, training=True)
        else:
            out = model(to_mace(batch, model), training=True, compute_force=True, compute_stress=True)
        loss = combine(loss_sums(out, batch, cfg), loss_counts(batch.natoms.tolist()), cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i >= warmup:
            n_atoms += int(batch.natoms.sum())
            n_structs += len(idx)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    return {
        "impl": impl,
        "batch_size": batch_size,
        "steps": steps,
        "seconds_per_step": dt / steps,
        "structures_per_s": n_structs / dt,
        "atoms_per_s": n_atoms / dt,
        "peak_mem_GB": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
    }


def peak_memory_GB(device: torch.device) -> float:
    """Peak GPU memory of this size, or the process's peak resident memory on the CPU, which
    is the peak of this size because the sizes ascend."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # bytes on macOS, kB on Linux


def _infer(model, device, dtype, backend, threads, calls, max_seconds, repeats) -> None:
    from ase.build import bulk

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    device = torch.device(device)
    calc = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend)
    for n in repeats:
        atoms = bulk("Cu", cubic=True).repeat(n)
        atoms.rattle(0.05, seed=0)
        rec = {
            "device": str(device),
            "dtype": dtype,
            "backend": calc.model.config.backend,
            "threads": threads,
        }
        rec["atoms"] = len(atoms)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        try:
            batch = calc._batch(atoms)
            rec["edges"] = batch.edge_index.shape[1]
            rec["graph_s"] = time.perf_counter() - t0
            calc.calculate(atoms, ["energy", "forces", "stress"])
            rec["first_call_s"] = time.perf_counter() - t0
            times = []
            for _ in range(calls if rec["first_call_s"] < max_seconds else 0):
                t0 = time.perf_counter()
                calc.calculate(atoms, ["energy", "forces", "stress"])
                times.append(time.perf_counter() - t0)
            s = float(np.median(times)) if times else rec["first_call_s"]
            rec["seconds_per_call"] = s
            rec["us_per_atom_call"] = s / len(atoms) * 1e6
            rec["peak_mem_GB"] = peak_memory_GB(device)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            rec["error"] = "oom"
        print(json.dumps(rec), flush=True)
        if "error" in rec or rec["first_call_s"] >= max_seconds:
            break


def run_infer(args: argparse.Namespace) -> None:
    for dtype in args.dtypes:
        for backend in args.backends:
            job = (
                args.model,
                args.device,
                dtype,
                backend,
                args.threads,
                args.calls,
                args.max_seconds,
                args.repeats,
            )
            spawn(_infer, [job], check=False)  # a process killed for memory still leaves the others


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", default="train", choices=["train", "infer"])
    p.add_argument("--frames", default=None, help="train: Frames HDF5 with neighbour lists")
    p.add_argument("--impl", nargs="+", default=["ours", "ours-cueq", "mace-e3nn", "mace-cueq"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile", action="store_true", help="print the most expensive ops of our model")
    p.add_argument("--model", default=None, help="infer: weights file or checkpoint")
    p.add_argument("--dtypes", nargs="+", default=["float32", "float64"], help="infer")
    p.add_argument("--backends", nargs="+", default=["e3nn"], choices=["e3nn", "cueq"], help="infer")
    p.add_argument("--threads", type=int, default=1, help="infer: torch threads")
    p.add_argument("--calls", type=int, default=5, help="infer: timed calls after the first, per size")
    p.add_argument("--max-seconds", type=float, default=120.0, help="infer: stop after a slower first call")
    p.add_argument(
        "--repeats", nargs="+", type=int, default=list(INFER_REPEATS), help="infer: cubic cells per axis"
    )


def run(args: argparse.Namespace) -> None:
    if args.mode == "infer":
        run_infer(args)
        return
    if args.frames is None:
        raise SystemExit("bench-speed needs --frames")
    device = torch.device(args.device)
    frames = Frames.load(args.frames)
    print(
        json.dumps(
            {
                "frames": len(frames),
                "mean_atoms": float(frames.natoms.mean()),
                "mean_edges": float(frames.nedges.mean()),
            }
        ),
        flush=True,
    )
    for impl in args.impl:
        try:
            print(
                json.dumps(benchmark(frames, impl, args.batch_size, args.steps, args.warmup, device)),
                flush=True,
            )
        except Exception as exc:
            print(json.dumps({"impl": impl, "error": repr(exc)}), flush=True)
    if args.profile:
        for backend in ("e3nn", "cueq"):
            try:
                profile_ours(frames, args.batch_size, device, backend)
            except Exception as exc:
                print(json.dumps({"profile": backend, "error": repr(exc)}), flush=True)


def profile_ours(frames: Frames, batch_size: int, device: torch.device, backend: str, steps: int = 5) -> None:
    from torch.profiler import ProfilerActivity, profile

    torch.set_default_dtype(torch.float32)
    print(f"== profile of our model, backend {backend}", flush=True)
    model = MACE(MACEConfig(backend=backend)).to(device)
    cfg = LossConfig()
    rng = np.random.default_rng(1)
    batches = [
        collate(frames, np.sort(rng.choice(len(frames), batch_size, replace=False))).to(
            device=device, dtype=torch.float32
        )
        for _ in range(steps + 2)
    ]

    def step(batch):
        out = model(batch, compute_force=True, compute_stress=True, training=True)
        combine(loss_sums(out, batch, cfg), loss_counts(batch.natoms.tolist()), cfg).backward()

    for b in batches[:2]:
        step(b)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities) as prof:
        for b in batches[2:]:
            step(b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=15), flush=True)
