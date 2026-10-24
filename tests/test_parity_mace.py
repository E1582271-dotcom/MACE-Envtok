"""Our MACE against the reference mace-torch implementation, with shared weights."""

import numpy as np
import pytest
import torch
from e3nn import o3

from envtok_mace.data.batch import collate
from envtok_mace.model.cg import u_matrix
from envtok_mace.model.mace import MACEConfig

mace_compat = pytest.importorskip("envtok_mace.model.mace_compat")
pytest.importorskip("mace")

KEYS = ("energy", "forces", "stress")


@pytest.fixture
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _compare(ours, ref, batch, dtype):
    out = ours(batch.to(dtype=dtype), compute_force=True, compute_stress=True)
    exp = ref(mace_compat.mace_input_dict(batch, ref), compute_force=True, compute_stress=True)
    return {k: (out[k].detach() - exp[k].detach()).abs().max().item() for k in KEYS}


@pytest.mark.parametrize("ir_out", ["0e", "1o"])
@pytest.mark.parametrize("nu", [1, 2, 3])
def test_coupling_tensors_match_mace(ir_out, nu, float64):
    from mace.tools.cg import U_matrix_real

    irreps = o3.Irreps("1x0e+1x1o+1x2e+1x3o")
    ours = u_matrix(irreps, ir_out, nu, dtype=torch.float64)
    ref = U_matrix_real(
        irreps_in=irreps, irreps_out=o3.Irreps(ir_out), correlation=nu, use_cueq_cg=False, dtype=torch.float64
    )[-1]
    torch.testing.assert_close(ours, ref.reshape(ours.shape), rtol=0, atol=0)


def test_random_init_parity_fp64(val40, float64):
    torch.manual_seed(0)
    ref = mace_compat.build_reference_mace(MACEConfig())
    ours = mace_compat.from_mace(ref)
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in ref.parameters())
    diff = _compare(ours, ref, collate(val40, np.arange(8)), torch.float64)
    assert max(diff.values()) < 1e-10, diff


def test_published_envtok_model_parity_fp64(val40, envtok_model_path, float64):
    ref = torch.load(envtok_model_path, map_location="cpu", weights_only=False).to(torch.float64)
    ours = mace_compat.from_mace(ref)
    diff = _compare(ours, ref, collate(val40, np.arange(len(val40))), torch.float64)
    assert max(diff.values()) < 1e-10, diff


def test_published_envtok_model_parity_fp32(val40, envtok_model_path):
    ref = torch.load(envtok_model_path, map_location="cpu", weights_only=False).to(torch.float32)
    ours = mace_compat.from_mace(ref)
    batch = collate(val40, np.arange(len(val40)))
    diff = _compare(ours, ref, batch, torch.float32)
    # float32 summation-order noise only (fp64 agrees to ~1e-13)
    assert diff["forces"] < 2e-4 and diff["stress"] < 1e-6, diff
    assert diff["energy"] / batch.natoms.min().item() < 1e-5, diff
