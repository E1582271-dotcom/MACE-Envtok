"""Write a stratified random subset of a Frames file (e.g. the fixed 10% of the
validation set used during training), so training processes load only that."""

from __future__ import annotations

import argparse

import numpy as np

from .frames import Frames
from .samplers import stratified_subset


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--fraction", type=float, required=True, help="fraction of every sub-dataset to keep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--complement",
        action="store_true",
        help="write every frame NOT in that subset (e.g. a held-out set never used for model selection)",
    )


def run(args: argparse.Namespace) -> None:
    frames = Frames.load(args.src)
    idx = stratified_subset(frames.source, args.fraction, args.seed)
    if args.complement:
        idx = np.setdiff1d(np.arange(len(frames)), idx)
    sub = frames.subset(idx)
    sub.attrs.update(
        subset_of=str(args.src),
        subset_fraction=args.fraction,
        subset_seed=args.seed,
        subset_complement=bool(args.complement),
    )
    sub.save(args.dst)
    print(f"wrote {len(sub):,} of {len(frames):,} frames -> {args.dst}", flush=True)
