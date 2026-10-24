"""Helpers for tests that run the real Trainer (single or multi-process, CPU/gloo)."""

import os
import socket
from pathlib import Path

import torch

FIXTURE = str(Path(__file__).parent / "fixtures" / "omat24_val_40.h5")


def tiny_config(out_dir, **overrides):
    from envtok_mace.train.trainer import TrainConfig

    base = dict(
        train_file=FIXTURE,
        val_file=FIXTURE,
        out_dir=str(out_dir),
        device="cpu",
        dtype="float64",
        num_workers=0,
        batch_size=2,
        accumulation=1,
        val_batch_size=8,
        val_fraction=1.0,
        max_epochs=5,
        log_every=1000,
        model={"num_channels": 8, "radial_mlp": [8, 8]},
    )
    base.update(overrides)
    return TrainConfig(**base)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, cfg_kwargs):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    torch.set_num_threads(1)
    from envtok_mace.train.dist import cleanup, init_distributed
    from envtok_mace.train.trainer import Trainer

    env = init_distributed("cpu")
    try:
        Trainer(tiny_config(**cfg_kwargs), env).fit()
    finally:
        cleanup()


def run_training(world: int, **cfg_kwargs):
    if world == 1:
        from envtok_mace.train.dist import DistEnv
        from envtok_mace.train.trainer import Trainer

        return Trainer(tiny_config(**cfg_kwargs), DistEnv()).fit()
    torch.multiprocessing.spawn(_worker, args=(world, free_port(), cfg_kwargs), nprocs=world, join=True)
