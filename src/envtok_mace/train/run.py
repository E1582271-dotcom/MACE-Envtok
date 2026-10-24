"""`envtok-mace train`: one process per GPU, started by mpiexec/torchrun (or alone)."""

from __future__ import annotations

import argparse
import json
import os

from .dist import cleanup, init_distributed
from .trainer import TrainConfig, Trainer


def _parse_walltime(text: str | None) -> float | None:
    if not text:
        return None
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return 3600 * h + 60 * m + s


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="YAML file with TrainConfig fields")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config fields, values parsed as JSON when possible",
    )
    p.add_argument(
        "--walltime",
        default=os.environ.get("ENVTOK_WALLTIME"),
        help="job walltime HH:MM:SS; the run checkpoints and exits shortly before it",
    )


def run(args: argparse.Namespace) -> None:
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    cfg = TrainConfig.from_yaml(args.config, overrides)
    walltime = _parse_walltime(args.walltime)
    if walltime is not None:
        cfg.walltime_seconds = walltime
    env = init_distributed(cfg.device)
    try:
        Trainer(cfg, env).fit()
    finally:
        cleanup()
