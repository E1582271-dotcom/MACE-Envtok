"""Add radius-graph neighbour lists to a Frames file.

Neighbour lists are computed once here (CPU, in parallel) and stored with the
structures, so training never builds graphs on the fly.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from .frames import Frames, _ptr
from .neighbors import radius_graph

_FRAMES: Frames | None = None
_CUTOFF: float = 0.0


def _init_worker(path: str, cutoff: float) -> None:
    global _FRAMES, _CUTOFF
    _FRAMES = Frames.load(path)
    _CUTOFF = cutoff


def _graph_chunk(bounds: tuple[int, int]):
    start, stop = bounds
    counts, idx, shifts = [], [], []
    for k in range(start, stop):
        f = _FRAMES.get(k)
        ei, us = radius_graph(f["positions"], f["cell"], f["pbc"], _CUTOFF)
        counts.append(ei.shape[1])
        idx.append(ei)
        shifts.append(us)
    return start, np.array(counts, dtype=np.int64), np.concatenate(idx, axis=1), np.concatenate(shifts)


def build_graphs(
    src: str | Path, dst: str | Path, cutoff: float, workers: int = 1, chunk: int = 2000
) -> Frames:
    t0 = time.time()
    frames = Frames.load(src)
    n = len(frames)
    bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    results = {}
    if workers > 1:
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(str(src), cutoff)) as pool:
            for i, (start, c, ei, us) in enumerate(pool.imap_unordered(_graph_chunk, bounds), 1):
                results[start] = (c, ei, us)
                if i % max(1, len(bounds) // 20) == 0 or i == len(bounds):
                    print(f"  {i}/{len(bounds)} chunks ({time.time() - t0:.0f} s)", flush=True)
    else:
        global _FRAMES, _CUTOFF
        _FRAMES, _CUTOFF = frames, cutoff
        for b in bounds:
            start, c, ei, us = _graph_chunk(b)
            results[start] = (c, ei, us)

    ordered = [results[s] for s, _ in bounds]
    frames.edge_ptr = _ptr(np.concatenate([r[0] for r in ordered]))
    frames.edge_index = np.concatenate([r[1] for r in ordered], axis=1)
    frames.unit_shifts = np.concatenate([r[2] for r in ordered])
    avg_nb = frames.edge_ptr[-1] / frames.atom_ptr[-1]
    frames.attrs.update(r_max=float(cutoff), avg_num_neighbors=float(avg_nb))
    frames.save(dst)
    print(
        f"wrote {n:,} graphs, {int(frames.edge_ptr[-1]):,} edges, "
        f"avg_num_neighbors {avg_nb:.4f} -> {dst} ({time.time() - t0:.0f} s)",
        flush=True,
    )
    return frames


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--src", required=True, help="Frames HDF5 written by `extract`")
    p.add_argument("--dst", required=True, help="output Frames HDF5 with neighbour lists")
    p.add_argument("--r-max", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk", type=int, default=2000)


def run(args: argparse.Namespace) -> None:
    build_graphs(args.src, args.dst, args.r_max, args.workers, args.chunk)
