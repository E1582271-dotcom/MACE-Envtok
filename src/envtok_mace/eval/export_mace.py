"""Write a model as a mace-torch file, which needs mace-torch but not this package.

    envtok-mace export-mace --src mace-envtok-1.0.0.pt --dst mace-envtok.model

    from mace.calculators import MACECalculator
    atoms.calc = MACECalculator(model_paths="mace-envtok.model", device="cpu", default_dtype="float32")

mace-torch compiles the file for LAMMPS:

    python -m mace.cli.create_lammps_model mace-envtok.model

In float32, the precision of the training and of the OMat24 and MatPES errors, the file and
this package differ only by summation order. In float64, the precision of the materials
benchmarks, a model trained in float32 differs a little more, up to 7e-7 eV/A on the test
frames: it holds the exponents of the ZBL screening function rounded to float32, and
mace-torch has them in its code, so a file cannot carry them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..model.mace_compat import to_mace
from .export import sha256
from .run_eval import load_model


def export_mace(src: str, dst: str) -> dict:
    model = load_model(src, torch.device("cpu"), torch.float64, "e3nn")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_mace(model), dst)
    return {"file": str(dst), "sha256": sha256(dst), "bytes": Path(dst).stat().st_size}


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--src", required=True, help="weights file or training checkpoint")
    p.add_argument("--dst", required=True, help="mace-torch file to write (.model)")


def run(args: argparse.Namespace) -> None:
    print(json.dumps(export_mace(args.src, args.dst), indent=1))
