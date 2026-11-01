"""The inference benchmark reports each size, and stops at the first size that is too slow."""

import json

import torch

from envtok_mace.bench.speed import _infer
from envtok_mace.model.mace import MACE, MACEConfig


def _weights(tmp_path):
    torch.manual_seed(0)
    model = MACE(MACEConfig())
    path = tmp_path / "weights.pt"
    torch.save({"model_config": model.config.to_dict(), "model": model.state_dict()}, path)
    return str(path)


def _records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_reports_every_size(tmp_path, capsys):
    _infer(_weights(tmp_path), "cpu", "float32", "e3nn", 1, 2, 1e6, (1, 2))
    first, second = _records(capsys)
    assert (first["atoms"], second["atoms"]) == (4, 32)
    assert first["edges"] > 0 and second["edges"] > first["edges"]
    assert first["us_per_atom_call"] == first["seconds_per_call"] / 4 * 1e6
    assert second["peak_mem_GB"] >= first["peak_mem_GB"] > 0


def test_stops_after_a_slow_first_call(tmp_path, capsys):
    _infer(_weights(tmp_path), "cpu", "float32", "e3nn", 1, 2, 0.0, (1, 2))
    records = _records(capsys)
    assert [r["atoms"] for r in records] == [4]
    assert records[0]["seconds_per_call"] == records[0]["first_call_s"]
