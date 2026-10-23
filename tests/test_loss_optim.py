import argparse

import numpy as np
import pytest
import torch

from envtok_mace.data.batch import collate
from envtok_mace.model.mace import MACE, MACEConfig
from envtok_mace.train.loss import LossConfig, combine, loss_counts, loss_sums
from envtok_mace.train.optim import EMA, OptimConfig, build_optimizer, decays


class _RefBatch:
    """Minimal stand-in for mace's torch_geometric Batch."""

    def __init__(self, batch):
        g = batch.num_graphs
        self.ptr = torch.zeros(g + 1, dtype=torch.long)
        self.ptr[1:] = torch.cumsum(batch.natoms, 0)
        self.energy, self.forces, self.stress = batch.energy, batch.forces, batch.stress
        self.energy_weight = torch.ones(g, dtype=batch.energy.dtype)
        self.forces_weight = torch.ones(g, dtype=batch.energy.dtype)
        self.stress_weight = torch.ones(g, dtype=batch.energy.dtype)

    def __getitem__(self, key):
        return getattr(self, key)


def _fake_prediction(batch, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "energy": batch.energy + scale * torch.randn(batch.energy.shape, generator=g, dtype=torch.float64),
        "forces": batch.forces + scale * torch.randn(batch.forces.shape, generator=g, dtype=torch.float64),
        "stress": batch.stress + 0.01 * scale * torch.randn(batch.stress.shape, generator=g, dtype=torch.float64),
    }


@pytest.mark.parametrize("scale", [0.001, 0.05, 2.0])  # inside, around and beyond the Huber delta
def test_loss_matches_mace_universal_loss(val40, scale):
    loss_mod = pytest.importorskip("mace.modules.loss")
    batch = collate(val40, np.arange(12))
    pred = _fake_prediction(batch, scale)
    cfg = LossConfig()
    ours = combine(loss_sums(pred, batch, cfg), loss_counts(batch.natoms), cfg)
    ref = loss_mod.UniversalLoss(energy_weight=1.0, forces_weight=10.0, stress_weight=10.0, huber_delta=0.01)
    theirs = ref.double()(_RefBatch(batch), pred, ddp=False)
    torch.testing.assert_close(ours, theirs, rtol=1e-12, atol=1e-15)


def test_large_force_atoms_use_smaller_delta(val40):
    batch = collate(val40, np.arange(2))
    batch.forces[0] = torch.tensor([250.0, 0.0, 0.0], dtype=torch.float64)
    pred = _fake_prediction(batch, 1.0)
    loss_mod = pytest.importorskip("mace.modules.loss")
    cfg = LossConfig()
    ours = combine(loss_sums(pred, batch, cfg), loss_counts(batch.natoms), cfg)
    theirs = loss_mod.UniversalLoss(1.0, 10.0, 10.0, 0.01).double()(_RefBatch(batch), pred, ddp=False)
    torch.testing.assert_close(ours, theirs, rtol=1e-12, atol=1e-15)


def test_split_sums_equal_whole_batch_mean(val40):
    """Summing per-part sums and dividing by total counts == loss of the whole batch."""
    cfg = LossConfig()
    whole = collate(val40, np.arange(10))
    pred_whole = _fake_prediction(whole, 0.05)
    full = combine(loss_sums(pred_whole, whole, cfg), loss_counts(whole.natoms), cfg)
    parts = [np.arange(0, 3), np.arange(3, 10)]
    counts = loss_counts(whole.natoms)
    total = 0.0
    a0 = 0
    for idx in parts:
        b = collate(val40, idx)
        a1 = a0 + int(b.natoms.sum())
        p = {"energy": pred_whole["energy"][idx], "forces": pred_whole["forces"][a0:a1],
             "stress": pred_whole["stress"][idx]}
        total = total + combine(loss_sums(p, b, cfg), counts, cfg)
        a0 = a1
    torch.testing.assert_close(total, full, rtol=1e-12, atol=1e-15)


def test_weight_decay_groups_match_mace():
    pytest.importorskip("mace")
    from mace.tools.scripts_utils import get_params_options

    from envtok_mace.model.mace_compat import _weight_map, build_reference_mace

    cfg = MACEConfig(num_channels=8, radial_mlp=(8, 8))
    ref = build_reference_mace(cfg)
    ours = MACE(cfg)
    args = argparse.Namespace(weight_decay=1e-6, lr=1e-3, amsgrad=True, beta=0.9)
    groups = get_params_options(args, ref)["params"]
    ref_names = {id(p): n for n, p in ref.named_parameters()}
    ref_decay = {ref_names[id(p)] for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    mapping = _weight_map(ours)
    our_decay = {mapping[n] for n, _ in ours.named_parameters() if decays(n)}
    assert our_decay == ref_decay


def test_optimizer_covers_every_parameter_once():
    model = MACE(MACEConfig(num_channels=8, radial_mlp=(8, 8)))
    opt = build_optimizer(model, OptimConfig())
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))


def test_ema_warmup_and_swap():
    lin = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(0.0)
    ema = EMA(lin, decay=0.999)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update()  # d = min(0.999, 2/11)
    expected = 1.0 - 2.0 / 11.0
    assert ema.shadow[0][0, 0].item() == pytest.approx(expected)
    with ema.average_parameters():
        assert lin.weight[0, 0].item() == pytest.approx(expected)
    assert lin.weight[0, 0].item() == 1.0
    state = ema.state_dict()
    ema2 = EMA(lin, decay=0.5)
    ema2.load_state_dict(state)
    assert ema2.num_updates == 1 and ema2.decay == 0.999
