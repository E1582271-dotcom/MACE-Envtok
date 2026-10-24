"""The cuEquivariance backend gives the same model as the e3nn backend."""

import numpy as np
import pytest
import torch

from envtok_mace.data.batch import collate
from envtok_mace.model.backends import convert_state_dict
from envtok_mace.model.mace import MACE, MACEConfig

pytest.importorskip("cuequivariance_torch")


@pytest.fixture
def pair():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3)
    cfg = dict(num_channels=16, radial_mlp=(16, 16))
    ref = MACE(MACEConfig(**cfg, backend="e3nn"))
    cueq = MACE(MACEConfig(**cfg, backend="cueq", cueq_fallback=True))  # CPU: reference implementation
    cueq.load_state_dict(convert_state_dict(ref.state_dict(), ref, cueq))
    return ref, cueq


def test_same_parameter_count(pair):
    ref, cueq = pair
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in cueq.parameters())


def test_cueq_matches_e3nn(pair, val40):
    ref, cueq = pair
    batch = collate(val40, np.arange(10)).to(dtype=torch.float64)
    a = ref(batch, compute_stress=True)
    b = cueq(batch, compute_stress=True)
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(b[key], a[key], rtol=1e-9, atol=1e-11, msg=key)


def test_round_trip_conversion(pair):
    ref, cueq = pair
    back = MACE(ref.config)
    back.load_state_dict(convert_state_dict(cueq.state_dict(), cueq, back))
    for (k, v), (_, w) in zip(ref.state_dict().items(), back.state_dict().items()):
        assert torch.equal(v, w), k


def test_gradients_match(pair, val40):
    ref, cueq = pair
    batch = collate(val40, np.arange(4)).to(dtype=torch.float64)
    grads = []
    for model in (ref, cueq):
        model.zero_grad()
        out = model(batch, compute_stress=True, training=True)
        (out["energy"].sum() + out["forces"].pow(2).sum()).backward()
        grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters()})
    mapped = convert_state_dict(grads[0], ref, cueq)
    for name, g in grads[1].items():
        torch.testing.assert_close(g, mapped[name], rtol=1e-8, atol=1e-10, msg=name)


def test_load_weights_tolerates_only_cuequivariance_constants(pair):
    """A checkpoint written where cuEquivariance's GPU kernels are installed lacks the
    constant buffers of its CPU implementation; loading it must still work, and must
    still fail on a missing parameter."""
    import re

    from envtok_mace.model.backends import load_weights

    _, cueq = pair
    state = cueq.state_dict()
    constants = [k for k in state if re.search(r"\.graph\.c\d+$", k)]
    assert constants, "the CPU build registers derived constants"
    stripped = {k: v for k, v in state.items() if k not in constants}
    fresh = MACE(MACEConfig(**{**cueq.config.to_dict()}))
    load_weights(fresh, stripped)
    for (name, a), b in zip(cueq.named_parameters(), fresh.parameters()):
        assert torch.equal(a, b), name
    # derived constants in a file are ignored: the module keeps the ones it built,
    # at the precision in use (a float32 copy would cost float64 runs accuracy)
    tampered = {k: (2 * v if "projection" in k else v) for k, v in state.items()}
    assert any("projection" in k for k in tampered)
    load_weights(fresh, tampered)
    for name, buf in fresh.named_buffers():
        if "projection" in name:
            assert torch.equal(buf, dict(cueq.named_buffers())[name]), name
    broken = dict(stripped)
    broken.pop(next(n for n, _ in cueq.named_parameters()))
    with pytest.raises(RuntimeError, match="missing"):
        load_weights(fresh, broken)
