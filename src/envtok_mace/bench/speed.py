"""Training-step throughput of our model and of the reference mace-torch model
(with and without cuEquivariance kernels) on the same batches, one GPU.

A step is forward (energy, forces, stress), loss, backward and an Adam update.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..data.batch import collate
from ..data.frames import Frames
from ..model.mace import MACE, MACEConfig
from ..train.loss import LossConfig, combine, loss_counts, loss_sums


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


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames", required=True, help="Frames HDF5 with neighbour lists")
    p.add_argument("--impl", nargs="+", default=["ours", "ours-cueq", "mace-e3nn", "mace-cueq"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile", action="store_true", help="print the most expensive ops of our model")


def run(args: argparse.Namespace) -> None:
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
