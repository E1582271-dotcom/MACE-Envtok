"""How structures are grouped into batches and spread over ranks.

Training: every epoch the training set is shuffled with a seed that depends only
on (seed, epoch) and cut into global batches. Inside each global batch the
structures are spread over S = world_size x accumulation equally sized slots so
that every slot gets a similar total size (edges + atoms), and every rank a
similar amount of work per step. The members of a global batch are
the same as with plain random batching; only their placement changes, so with
globally normalised losses the gradient is unchanged.
"""

from __future__ import annotations

import numpy as np


def snake_slots(n: int, num_slots: int) -> np.ndarray:
    """Slot index for each of n items dealt in snake order."""
    pattern = np.concatenate([np.arange(num_slots), np.arange(num_slots)[::-1]])
    return np.resize(pattern, n)


def balanced_slots(cost: np.ndarray, num_slots: int) -> np.ndarray:
    """Assign items to ``num_slots`` equally sized slots with similar total cost.

    Greedy "largest first": take items from the most to the least expensive and
    give each to the least loaded slot that still has room.
    """
    n = len(cost)
    capacity = n // num_slots
    assert capacity * num_slots == n, "items must split evenly over slots"
    order = np.argsort(-cost, kind="stable")
    load = np.zeros(num_slots)
    filled = np.zeros(num_slots, dtype=np.int64)
    slot = np.empty(n, dtype=np.int64)
    big = np.inf
    for i in order:
        s = int(np.argmin(np.where(filled < capacity, load, big)))
        slot[i] = s
        load[s] += cost[i]
        filled[s] += 1
    return slot


class TrainPlan:
    def __init__(self, cost: np.ndarray, per_rank_batch: int, world_size: int = 1,
                 accumulation: int = 1, seed: int = 0, balance: bool = True):
        self.cost = np.asarray(cost)
        self.per_rank_batch = per_rank_batch
        self.world_size = world_size
        self.accumulation = accumulation
        self.seed = seed
        self.balance = balance

    @property
    def global_batch(self) -> int:
        return self.per_rank_batch * self.world_size * self.accumulation

    @property
    def steps_per_epoch(self) -> int:
        return len(self.cost) // self.global_batch

    def global_batches(self, epoch: int) -> np.ndarray:
        """[steps, global_batch] structure indices of this epoch (identical on all ranks)."""
        rng = np.random.default_rng([self.seed, epoch])
        perm = rng.permutation(len(self.cost))
        n = self.steps_per_epoch * self.global_batch
        return perm[:n].reshape(self.steps_per_epoch, self.global_batch)

    def rank_steps(self, epoch: int, rank: int) -> list[list[np.ndarray]]:
        """For each optimiser step, the micro-batches (index arrays) of ``rank``."""
        slots = self.world_size * self.accumulation
        steps = []
        for members in self.global_batches(epoch):
            if self.balance:
                owner = balanced_slots(self.cost[members], slots)
                per_slot = [members[owner == s] for s in range(slots)]
            else:
                per_slot = np.split(members, slots)
            # slot = micro * world + rank
            steps.append([per_slot[m * self.world_size + rank] for m in range(self.accumulation)])
        return steps


def eval_shards(indices: np.ndarray, cost: np.ndarray, world_size: int, rank: int,
                batch_size: int, max_cost: float | None = None) -> list[np.ndarray]:
    """Split evaluation structures over ranks (no padding, no duplicates) and into
    batches of similar size: at most ``batch_size`` structures and, if given, at
    most ``max_cost`` total cost (e.g. edges) per batch, so batches of the largest
    structures stay small."""
    indices = np.asarray(indices)
    order = indices[np.argsort(-cost[indices], kind="stable")]
    mine = order[snake_slots(len(order), world_size) == rank]
    batches, start, total = [], 0, 0.0
    for i, k in enumerate(mine):
        full = i - start >= batch_size or (max_cost is not None and i > start and total + cost[k] > max_cost)
        if full:
            batches.append(mine[start:i])
            start, total = i, 0.0
        total += cost[k]
    if start < len(mine):
        batches.append(mine[start:])
    return batches


def stratified_subset(groups: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """A fixed random fraction of every group (e.g. every OMat24 sub-dataset), sorted."""
    rng = np.random.default_rng(seed)
    picked = []
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        k = max(1, int(round(fraction * len(members))))
        picked.append(rng.choice(members, size=k, replace=False))
    return np.sort(np.concatenate(picked))
