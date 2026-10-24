"""Process-group setup and the few collective operations the trainer needs.

Rank information is read from whichever launcher started the processes:
torchrun (RANK/LOCAL_RANK/WORLD_SIZE), Cray PALS mpiexec (PALS_*/PMI_*) or
Open MPI (OMPI_COMM_WORLD_*). MASTER_ADDR/MASTER_PORT must be set by the job
script for multi-node runs.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        if os.environ.get(name, "") != "":
            return int(os.environ[name])
    return default


def init_distributed(
    device: str = "cuda", backend: str | None = None, timeout_minutes: int | None = None
) -> DistEnv:
    if timeout_minutes is None:
        timeout_minutes = int(os.environ.get("ENVTOK_PG_TIMEOUT_MIN", "30"))
    rank = _env_int(("RANK", "PMI_RANK", "PALS_RANKID", "OMPI_COMM_WORLD_RANK"), 0)
    world = _env_int(("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE"), 1)
    local = _env_int(("LOCAL_RANK", "PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"), 0)
    if device == "cuda" and not torch.cuda.is_available():
        # Never fall back silently: a CPU/gloo rank among NCCL ranks hangs the whole job.
        raise RuntimeError(
            f"rank {rank} on {socket.gethostname()}: device 'cuda' requested but no GPU is "
            f"visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
        )
    use_cuda = device == "cuda"
    if use_cuda:
        # The job sees only its own GPUs on each node (PBS cgroups); ranks on a node
        # take them in order of their local rank.
        n_visible = torch.cuda.device_count()
        if local >= n_visible:
            raise RuntimeError(
                f"local rank {local} but only {n_visible} visible GPU(s) on {socket.gethostname()}"
            )
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
        print(
            f"rank {rank}/{world} on {socket.gethostname()}: local rank {local}, "
            f"GPU {torch.cuda.get_device_name(local)} (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})",
            flush=True,
        )
    else:
        dev = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_PORT", "29500")
        if "MASTER_ADDR" not in os.environ:
            raise RuntimeError("MASTER_ADDR is not set; the job script must export it")
        os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)
        dist.init_process_group(
            backend or ("nccl" if use_cuda else "gloo"),
            init_method="env://",
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=timeout_minutes),
        )
    return DistEnv(rank=rank, local_rank=local, world_size=world, device=dev)


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_gather_objects(obj) -> list:
    if dist.is_initialized() and dist.get_world_size() > 1:
        out = [None] * dist.get_world_size()
        dist.all_gather_object(out, obj)
        return out
    return [obj]


def broadcast_flag(flag: bool, device: torch.device) -> bool:
    """Rank 0's decision, made identical on every rank."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        t = torch.tensor([1.0 if flag else 0.0], device=device)
        dist.broadcast(t, src=0)
        return bool(t.item() > 0.5)
    return flag


def barrier() -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
