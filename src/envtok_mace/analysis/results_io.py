"""The benchmark summaries in ``results/``, loaded under one set of model names.

The files were written by different commands, which name the models differently:
the eval JSONs say ``ours_envtok`` and ``authors_rdm1``..``4``, every other file
``ours`` and ``rdm1``..``4``, and the stability summary keeps our speed-matched
runs as ``ours@speed_of_authors_envtok`` next to the models. ``load`` returns each
summary under the names the tables and figures use; nothing is recomputed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SEEDS = ("rdm1", "rdm2", "rdm3", "rdm4")
MODELS = ("ours", "authors_envtok", "authors_direct", "mace_omat0", *SEEDS)
# our MD runs cut to the wall-clock budget the authors' slower model had (stability.py)
SPEED_MATCHED = "ours@speed_of_authors_envtok"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def read_csv(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def _eval_name(label: str) -> str:
    if label == "ours_envtok":
        return "ours"
    return label.removeprefix("authors_") if label.startswith("authors_rdm") else label


def load(results: Path) -> dict:
    """The headline summaries: ``val`` (OMat24 held-out), ``matpes``, ``matcalc``,
    ``stability`` and ``neb``, each keyed by model name."""
    results = Path(results)
    val = read_json(results / "eval/val90_heldout_all.json")["results"]
    # MACE-OMAT-0 failed in the first run and was scored separately
    val.update(read_json(results / "eval/val90_heldout_omat0.json")["results"])
    return {
        "val": {_eval_name(k): v for k, v in val.items()},
        "matpes": read_json(results / "matpes/tables/matpes_tables.json")["metrics"],
        "matcalc": read_json(results / "matcalc/matcalc_summary.json")["quantities"],
        "stability": read_json(results / "stability/stability_summary.json"),
        "neb": read_json(results / "nebdft2k/nebdft2k_summary.json"),
    }
