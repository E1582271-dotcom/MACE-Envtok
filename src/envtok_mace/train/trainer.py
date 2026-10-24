"""Training loop: data-parallel over GPUs (DDP), optional gradient accumulation,
EMA weights, plateau LR schedule, early stopping and exact resume.

One optimiser step uses a global batch of
    batch_size (per rank and micro-batch) x world_size x accumulation
structures. Loss terms are normalised by element counts summed over the whole
global batch, so the update is the same however the batch is split.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data.batch import collate
from ..data.frames import Frames
from ..data.samplers import TrainPlan, eval_shards, stratified_subset
from ..eval.metrics import evaluate
from ..model.mace import MACE, MACEConfig
from .checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from .dist import DistEnv, all_gather_objects, all_reduce_sum, barrier, broadcast_flag
from .loss import LossConfig, combine, loss_sums
from .optim import EMA, OptimConfig, build_optimizer, build_scheduler
from .tracking import Tracker


@dataclass
class TrainConfig:
    train_file: str
    val_file: str
    out_dir: str
    run_name: str = "envtok"
    model: dict = field(default_factory=dict)   # MACEConfig fields to override
    loss: dict = field(default_factory=dict)    # LossConfig fields to override
    optim: dict = field(default_factory=dict)   # OptimConfig fields to override
    batch_size: int = 128           # structures per rank per micro-batch
    accumulation: int = 1           # micro-batches per optimiser step (ignored if global_batch is set)
    global_batch: int | None = None # if set: accumulation = global_batch / (batch_size x world_size),
                                    # so the run is the same on 8, 4, 2 or 1 GPUs
    val_batch_size: int = 64
    val_fraction: float = 0.1       # stratified by sub-dataset, fixed for the whole run
    val_seed: int = 0
    max_epochs: int = 1000
    patience: int = 40              # epochs without improvement before stopping
    max_steps: int | None = None    # optional cap on optimiser steps (pilots, tests)
    seed: int = 1
    dtype: str = "float32"
    device: str = "cuda"
    num_workers: int = 8
    balance: bool = True
    log_every: int = 50
    archive_every: int = 25         # keep a copy of the checkpoint every N epochs
    walltime_seconds: float | None = None
    walltime_margin_seconds: float = 1200.0
    nan_tolerance: int = 20
    use_wandb: bool = False
    max_train_frames: int | None = None
    interrupt_after_steps: int | None = None  # behave as if walltime ran out (tests of resume)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.update(overrides or {})
        for key in ("train_file", "val_file", "out_dir"):  # allow ${ENVTOK_DATA} etc. in paths
            if isinstance(data.get(key), str):
                data[key] = os.path.expandvars(data[key])
        return cls(**data)


class _Indices(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


class Trainer:
    def __init__(self, cfg: TrainConfig, env: DistEnv):
        self.cfg, self.env = cfg, env
        self.dtype = getattr(torch, cfg.dtype)
        torch.set_default_dtype(self.dtype)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.train_frames = Frames.load(cfg.train_file)
        if cfg.max_train_frames:
            self.train_frames = self.train_frames.subset(np.arange(min(cfg.max_train_frames, len(self.train_frames))))
        val_all = Frames.load(cfg.val_file)
        self.val_frames = val_all.subset(stratified_subset(val_all.source, cfg.val_fraction, cfg.val_seed))
        del val_all

        self.model = MACE(MACEConfig(**cfg.model)).to(device=env.device, dtype=self.dtype)
        self.net = self.model
        if env.distributed:
            self.net = DDP(self.model, device_ids=[env.local_rank] if env.device.type == "cuda" else None,
                           broadcast_buffers=False, gradient_as_bucket_view=True)
        self.loss_cfg = LossConfig(**cfg.loss)
        self.optim_cfg = OptimConfig(**cfg.optim)
        self.optimizer = build_optimizer(self.model, self.optim_cfg)
        self.scheduler = build_scheduler(self.optimizer, self.optim_cfg)
        self.ema = EMA(self.model, self.optim_cfg.ema_decay)

        cost = self.train_frames.nedges + self.train_frames.natoms
        self.accumulation = self._accumulation()
        self.plan = TrainPlan(cost, cfg.batch_size, env.world_size, self.accumulation, cfg.seed, cfg.balance)
        if self.plan.steps_per_epoch == 0:
            raise ValueError("training set is smaller than one global batch")
        self.val_batches = eval_shards(np.arange(len(self.val_frames)), self.val_frames.nedges,
                                       env.world_size, env.rank, cfg.val_batch_size)

        self.epoch = 0
        self.step_in_epoch = 0
        self.global_step = 0
        self.best_val = math.inf
        self.bad_epochs = 0
        self.nan_count = 0
        self.out = Path(cfg.out_dir)
        self.t_start = time.time()
        self.steps_this_run = 0
        self.tracker = Tracker(self.out, env.is_main, self._run_info(), cfg.use_wandb, cfg.run_name)

    # ------------------------------------------------------------------ utils
    def _accumulation(self) -> int:
        cfg, world = self.cfg, self.env.world_size
        if cfg.global_batch is None:
            return cfg.accumulation
        per_step = cfg.batch_size * world
        if cfg.global_batch % per_step:
            raise ValueError(f"global_batch {cfg.global_batch} is not a multiple of "
                             f"batch_size {cfg.batch_size} x {world} ranks")
        return cfg.global_batch // per_step

    def _run_info(self) -> dict:
        return {
            "train": asdict(self.cfg),
            "model": self.model.config.to_dict(),
            "world_size": self.env.world_size,
            "accumulation": self.accumulation,
            "global_batch": self.plan.global_batch,
            "steps_per_epoch": self.plan.steps_per_epoch,
            "n_train": len(self.train_frames),
            "n_val": len(self.val_frames),
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
            "git_commit": _git_commit(),
            "torch": torch.__version__,
        }

    def log(self, msg: str) -> None:
        if self.env.is_main:
            print(msg, flush=True)

    def _time_left(self) -> float:
        if self.cfg.walltime_seconds is None:
            return math.inf
        return self.cfg.walltime_seconds - (time.time() - self.t_start)

    # ------------------------------------------------------------- checkpoints
    def _state(self) -> dict:
        return {
            "train_config": asdict(self.cfg),
            "model_config": self.model.config.to_dict(),
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "bad_epochs": self.bad_epochs,
            "nan_count": self.nan_count,
            "world_size": self.env.world_size,
        }

    def save_last(self) -> None:
        rngs = all_gather_objects(rng_state())
        if self.env.is_main:
            state = self._state()
            state["rng"] = rngs
            save_checkpoint(self.out / "last.pt", state)
        barrier()

    def save_best(self, metrics: dict) -> None:
        if not self.env.is_main:
            return
        with self.ema.average_parameters():
            weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        save_checkpoint(self.out / "best.pt", {"model_config": self.model.config.to_dict(), "model": weights,
                                               "epoch": self.epoch, "val": metrics})

    def maybe_resume(self) -> bool:
        path = self.out / "last.pt"
        if not path.exists():
            return False
        state = load_checkpoint(path, map_location=self.env.device)
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        for key in ("epoch", "step_in_epoch", "global_step", "best_val", "bad_epochs", "nan_count"):
            setattr(self, key, state[key])
        if state.get("world_size") == self.env.world_size and "rng" in state:
            set_rng_state(state["rng"][self.env.rank])
        self.log(f"resumed from {path}: epoch {self.epoch}, step {self.step_in_epoch}, "
                 f"global step {self.global_step}, best val {self.best_val:.6g}")
        return True

    # ---------------------------------------------------------------- training
    def _loader(self, microbatches: list[np.ndarray]):
        cuda = self.env.device.type == "cuda"
        kw = dict(batch_sampler=[list(m) for m in microbatches],
                  collate_fn=functools.partial(_collate_indices, self.train_frames),
                  num_workers=self.cfg.num_workers, pin_memory=cuda)
        if self.cfg.num_workers > 0:
            kw["prefetch_factor"] = 4
        return torch.utils.data.DataLoader(_Indices(len(self.train_frames)), **kw)

    def _global_counts(self, micro: list[np.ndarray]) -> dict[str, float]:
        natoms = self.train_frames.natoms
        local = torch.tensor([sum(len(m) for m in micro), sum(int(natoms[m].sum()) for m in micro)],
                             dtype=torch.float64, device=self.env.device)
        g, a = all_reduce_sum(local).tolist()
        return {"energy": g, "forces": 3.0 * a, "stress": 9.0 * g}

    def train_epoch(self) -> str | None:
        """Run the rest of the current epoch. Returns "walltime" or "max_steps" if stopped early."""
        steps = self.plan.rank_steps(self.epoch, self.env.rank)
        start = self.step_in_epoch
        loader = iter(self._loader([m for step in steps[start:] for m in step]))
        world = self.env.world_size
        params = [p for p in self.model.parameters() if p.requires_grad]
        window_t, window_structs, window_atoms = 0.0, 0, 0
        for s in range(start, len(steps)):
            micro = steps[s]
            counts = self._global_counts(micro)
            t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), dtype=torch.float64, device=self.env.device)
            for m in range(len(micro)):
                batch = next(loader).to(device=self.env.device, dtype=self.dtype)
                sync = m == len(micro) - 1
                ctx = self.net.no_sync() if (self.env.distributed and not sync) else contextlib.nullcontext()
                with ctx:
                    out = self.net(batch, compute_force=True, compute_stress=True, training=True)
                    loss = combine(loss_sums(out, batch, self.loss_cfg), counts, self.loss_cfg)
                    # DDP averages gradients over ranks; scale so the result is the global mean.
                    (loss * world).backward()
                step_loss += loss.detach().double()
            step_loss = all_reduce_sum(step_loss)
            if not torch.isfinite(step_loss):
                self.optimizer.zero_grad(set_to_none=True)
                self.nan_count += 1
                self.log(f"non-finite loss at global step {self.global_step}; skipped ({self.nan_count})")
                if self.nan_count >= self.cfg.nan_tolerance:
                    self.save_last()
                    raise FloatingPointError("too many non-finite losses")
                continue
            torch.nn.utils.clip_grad_norm_(params, self.optim_cfg.clip_grad)
            self.optimizer.step()
            self.ema.update()
            self.global_step += 1
            self.step_in_epoch = s + 1
            self.steps_this_run += 1

            if self.env.device.type == "cuda":
                torch.cuda.synchronize(self.env.device)
            dt = time.perf_counter() - t0
            window_t += dt
            window_structs += sum(len(m) for m in micro)
            window_atoms += sum(int(self.train_frames.natoms[m].sum()) for m in micro)
            if self.global_step % self.cfg.log_every == 0:
                times = all_gather_objects(window_t)
                n_structs = all_reduce_sum(torch.tensor(float(window_structs), device=self.env.device)).item()
                n_atoms = all_reduce_sum(torch.tensor(float(window_atoms), device=self.env.device)).item()
                if self.env.is_main:
                    t_max = max(times)
                    row = {
                        "epoch": self.epoch, "step": self.step_in_epoch, "global_step": self.global_step,
                        "lr": self.optimizer.param_groups[0]["lr"], "loss": step_loss.item(),
                        "structures_per_s": n_structs / t_max, "atoms_per_s": n_atoms / t_max,
                        "seconds_per_step": t_max / self.cfg.log_every,
                        "rank_time_max_over_mean": t_max / (sum(times) / len(times)),
                        "max_mem_GB": (torch.cuda.max_memory_allocated(self.env.device) / 1e9
                                       if self.env.device.type == "cuda" else 0.0),
                    }
                    self.tracker.log("steps", row)
                    print(json.dumps(row), flush=True)
                window_t, window_structs, window_atoms = 0.0, 0, 0
            if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                return "max_steps"
            if self.cfg.interrupt_after_steps and self.steps_this_run >= self.cfg.interrupt_after_steps:
                return "walltime"
            if self.global_step % 10 == 0 and broadcast_flag(
                    self._time_left() < self.cfg.walltime_margin_seconds, self.env.device):
                return "walltime"
        return None

    def validate(self) -> dict:
        with self.ema.average_parameters():
            return evaluate(self.model, self.val_frames, self.val_batches, self.env.device,
                            self.dtype, self.loss_cfg)

    def fit(self) -> str:
        if (self.out / "FINISHED").exists():
            self.log(f"{self.out}/FINISHED exists; nothing to do")
            return "finished"
        if not self.maybe_resume():
            metrics = self.validate()
            self.tracker.log("epochs", {"epoch": -1, "global_step": 0, "lr": self.optim_cfg.lr,
                                        "epoch_seconds": 0.0, "bad_epochs": 0, **metrics})
            self.log(f"initial validation: {json.dumps(metrics)}")
        reason = "max_epochs"
        while self.epoch < self.cfg.max_epochs:
            t_epoch = time.time()
            stop = self.train_epoch()
            if stop == "walltime":
                self.save_last()
                self.log(f"walltime almost used up; checkpoint saved at epoch {self.epoch}, "
                         f"step {self.step_in_epoch}")
                return "walltime"
            metrics = self.validate()
            self.scheduler.step(metrics["loss"])
            improved = metrics["loss"] < self.best_val
            if improved:
                self.best_val, self.bad_epochs = metrics["loss"], 0
                self.save_best(metrics)
            else:
                self.bad_epochs += 1
            row = {"epoch": self.epoch, "global_step": self.global_step,
                   "lr": self.optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - t_epoch,
                   "bad_epochs": self.bad_epochs, **metrics}
            self.tracker.log("epochs", row)
            self.log(json.dumps(row))
            self.epoch += 1
            self.step_in_epoch = 0
            self.save_last()
            if self.env.is_main and self.epoch % self.cfg.archive_every == 0:
                shutil.copyfile(self.out / "last.pt", self.out / f"epoch_{self.epoch:04d}.pt")
            if stop == "max_steps":
                reason = "max_steps"
                break
            if broadcast_flag(self.bad_epochs >= self.cfg.patience, self.env.device):
                reason = "early_stopping"
                break
        if self.env.is_main:
            (self.out / "FINISHED").write_text(f"{reason}\n")
        self.log(f"training finished ({reason}); best val loss {self.best_val:.6g}")
        self.tracker.close()
        return reason


def _collate_indices(frames: Frames, indices: list[int]):
    return collate(frames, np.asarray(indices, dtype=np.int64))


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"
