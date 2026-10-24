"""One table of every benchmark, grouped by how far the test data lie from training.

  ID        in distribution: OMat24 validation frames, generated like the training data
  shift     same kind of structures, different dataset and DFT settings: MatPES bulk,
            and near-equilibrium MatCalc properties
  OOD       outside the training distribution: MatPES non-bulk (low-dimensional and
            fragmented cells), high-energy states (softening), MD at up to 3,000 K and
            500 GPa, and NEB transition states

Every row is built from the result files in ``results/``; ``random`` is the mean
over the authors' four random-subset seeds (or their pooled runs, as the paper
reports them for MatCalc and MD), and each row says which way is better.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLUMNS = ("ours", "authors_envtok", "authors_direct", "random", "mace_omat0")
SEEDS = ("rdm1", "rdm2", "rdm3", "rdm4")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def rows_from_results(results: Path) -> list[dict]:
    rows = []

    def add(category, test, metric, unit, better, values):
        rows.append(
            {
                "category": category,
                "test": test,
                "metric": metric,
                "unit": unit,
                "better": better,
                **{c: values.get(c) for c in COLUMNS},
            }
        )

    # ---- ID: OMat24 validation (held-out 90 %)
    val = _load(results / "eval/val90_heldout_all.json")["results"]
    val.update(_load(results / "eval/val90_heldout_omat0.json")["results"])
    name = {
        "ours": "ours_envtok",
        "authors_envtok": "authors_envtok",
        "authors_direct": "authors_direct",
        "mace_omat0": "mace_omat0",
    }
    for key, metric, unit in (
        ("energy_mae_meV_per_atom", "energy MAE", "meV/atom"),
        ("force_mae_meV_per_A", "force MAE", "meV/A"),
        ("stress_mae_meV_per_A3", "stress MAE", "meV/A^3"),
    ):
        v = {c: val[n][key] for c, n in name.items()}
        v["random"] = float(np.mean([val[f"authors_{s}"][key] for s in SEEDS]))
        add("ID", "OMat24 validation (591,981 frames)", metric, unit, "lower", v)

    # ---- MatPES: bulk = dataset shift, non-bulk = OOD
    mp = _load(results / "matpes/tables/matpes_tables.json")["metrics"]["f_mae_comp"]["domains"]
    for dom, cat, test in (
        ("bulk", "shift", "MatPES bulk (242,319)"),
        ("non_bulk", "OOD", "MatPES non-bulk (80,766)"),
    ):
        for est, metric in (
            ("median", "force MAE per structure, median"),
            ("pooled_mean_MAE", "force MAE, pooled mean"),
        ):
            e = mp[dom]["estimators"][est]
            v = {c: 1e3 * e[c] for c in ("ours", "authors_envtok", "authors_direct", "mace_omat0")}
            v["random"] = 1e3 * float(np.mean([e[s] for s in SEEDS]))
            add(cat, test, metric, "meV/A", "lower", v)

    # ---- MatCalc: near-equilibrium properties = shift, high-energy states = OOD
    mc = _load(results / "matcalc/matcalc_summary.json")["quantities"]
    for key, stat, cat, test, metric, unit, better, scale in (
        ("eform", "mean", "shift", "MatCalc WBM (474)", "formation energy MAE", "meV/atom", "lower", 1e3),
        (
            "kvrh",
            "median",
            "shift",
            "MatCalc MP binaries (3,127)",
            "bulk modulus, median signed error",
            "%",
            "closer to 0",
            1,
        ),
        (
            "gvrh",
            "median",
            "shift",
            "MatCalc MP binaries (3,127)",
            "shear modulus, median signed error",
            "%",
            "closer to 0",
            1,
        ),
        (
            "cv_abs",
            "median",
            "shift",
            "MatCalc Alexandria (1,050)",
            "heat capacity, median abs error",
            "%",
            "lower",
            1,
        ),
        (
            "softening",
            "median",
            "OOD",
            "MatCalc WBM high-energy states (480)",
            "force softening, median",
            "%",
            "lower",
            1,
        ),
    ):
        m = mc[key]["models"]
        add(cat, test, metric, unit, better, {c: scale * m[c][stat] for c in COLUMNS})

    # ---- MD stability (OOD): the authors' runs at their own speed, ours at theirs
    st = _load(results / "stability/stability_summary.json")
    ours = st["ours@speed_of_authors_envtok"]
    for key, metric, unit, better in (
        ("mean_completion_pct", "mean completion", "%", "higher"),
        ("phys_div", "physically diverged runs", "% of runs", "lower"),
    ):
        get = (
            (lambda r: r["mean_completion_pct"])
            if key == "mean_completion_pct"
            else (lambda r: 100.0 * r["n_diverged"] / r["n_runs"])
        )
        v = {"ours": get(ours), "random": get(st["random"])}
        v.update({c: get(st[c]) for c in ("authors_envtok", "authors_direct", "mace_omat0")})
        add("OOD", "MD, NPT 300-3000 K, 0-500 GPa (80 runs)", metric, unit, better, v)

    # ---- NEB transition states (OOD)
    nb = _load(results / "nebdft2k/nebdft2k_summary.json")
    for split, metric in (
        ("without_U", "barrier MAE, hosts without Hubbard U"),
        ("with_U", "barrier MAE, hosts with Hubbard U"),
    ):
        v = {c: 1e3 * nb[c][split]["mae"] for c in COLUMNS}
        add("OOD", "nebDFT2k Li-ion migration (1,656 hops)", metric, "meV", "lower", v)
    return rows


def write(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keys = ["category", "test", "metric", "unit", "better", *COLUMNS]
    with (out / "id_ood_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4g}" if isinstance(r[k], float) else r[k]) for k in keys})
    lines = [
        "| Category | Test | Metric | Ours | Authors EnvTok | Authors DIRECT "
        "| Random (seeds) | MACE-OMAT-0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = " | ".join(f"{r[c]:.3g}" for c in COLUMNS)
        lines.append(
            f"| {r['category']} | {r['test']} | {r['metric']} ({r['unit']}, {r['better']}) | {cells} |"
        )
    (out / "id_ood_summary.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows -> {out}/id_ood_summary.csv and .md")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/tables")


def run(args: argparse.Namespace) -> None:
    write(rows_from_results(Path(args.results)), Path(args.out))
