from pathlib import Path

import pytest

import envtok_mace  # noqa: F401  (must come before e3nn is imported)

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]
ZENODO_MODELS = REPO / "local" / "zenodo" / "models" / "models" / "6.5M-atoms"


@pytest.fixture(scope="session")
def val40():
    from envtok_mace.data.frames import Frames

    return Frames.load(FIXTURES / "omat24_val_40.h5")


@pytest.fixture(scope="session")
def envtok_model_path():
    """Path of the authors' released EnvTok model, if it has been downloaded."""
    hits = sorted(ZENODO_MODELS.glob("*v14_T100*.model")) if ZENODO_MODELS.exists() else []
    if not hits:
        pytest.skip("authors' EnvTok model not downloaded (scripts/data/download_zenodo.sh)")
    return str(hits[0])


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """The trainer sets torch's default dtype; keep tests independent of each other."""
    import torch

    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)
