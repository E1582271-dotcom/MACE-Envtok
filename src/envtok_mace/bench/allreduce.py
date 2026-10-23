"""All-reduce bandwidth between ranks (run under mpiexec/torchrun)."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from ..train.dist import cleanup, init_distributed


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sizes-mb", type=float, nargs="+", default=[1, 27, 128])
    p.add_argument("--iters", type=int, default=20)


def run(args: argparse.Namespace) -> None:
    env = init_distributed("cuda")
    try:
        for mb in args.sizes_mb:
            x = torch.ones(int(mb * 1e6 / 4), device=env.device)
            for _ in range(3):
                dist.all_reduce(x)
            torch.cuda.synchronize(env.device)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                dist.all_reduce(x)
            torch.cuda.synchronize(env.device)
            dt = (time.perf_counter() - t0) / args.iters
            # ring all-reduce moves 2 (n-1)/n of the buffer per rank ("bus bandwidth")
            n = env.world_size
            busbw = 2 * (n - 1) / n * x.numel() * 4 / dt / 1e9
            if env.is_main:
                print(json.dumps({"world": n, "size_MB": mb, "ms": dt * 1e3, "bus_GBps": busbw}), flush=True)
    finally:
        cleanup()
