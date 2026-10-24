"""Every figure of the repository, drawn from the files in ``results/`` only.

    envtok-mace figures --results results --out results/figures

Each figure answers one question and is written as SVG (editable text), PDF and
a 300 dpi PNG. Colours, markers, sizes and fonts come from ``plots.style``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .style import (DOUBLE_COLUMN_IN, EDGE_PT, INK, MODELS, ONE_HALF_COLUMN_IN, SINGLE_COLUMN_IN,
                    color, line_color, style)

SEEDS = ("rdm1", "rdm2", "rdm3", "rdm4")
COMPARED = ("ours", "authors_envtok", "authors_direct", "random")
FIGURES: dict[str, callable] = {}


def figure(fn):
    FIGURES[fn.__name__] = fn
    return fn


# --------------------------------------------------------------------- helpers
def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def mark(ax, x, y, model, ms=5.5, z=3, **kw):
    spec = MODELS[model]
    ax.plot(x, y, spec["marker"], color=spec["color"], ms=ms, mec=spec["edge"], mew=EDGE_PT,
            zorder=z, linestyle="none", **kw)


def handle(model, line=True, label=None, marker=True):
    from matplotlib.lines import Line2D

    spec = MODELS[model]
    if spec["marker"] is None or not marker:
        if spec["marker"] is not None:
            return Line2D([], [], color=line_color(model), lw=1.25, label=label or spec["label"])
        return Line2D([], [], color=spec["color"], lw=0.9, ls=spec["linestyle"],
                      label=label or spec["label"])
    return Line2D([], [], color=line_color(model) if line else "none", lw=1.25,
                  marker=spec["marker"], ms=4.5, mfc=spec["color"], mec=spec["edge"], mew=EDGE_PT,
                  label=label or spec["label"])


def chip(ax, text):
    ax.text(1.0, 1.03, text, transform=ax.transAxes, fontsize=5.5, fontweight="bold",
            va="bottom", ha="right", color=INK["ink2"],
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc=INK["chip"], ec="none"))


def letter(ax, text, x=-0.02):
    ax.text(x, 1.03, text, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right", color=INK["ink"])


def reference(ax, value, label=True, vertical=True, text="MACE-OMAT-0", pos=None, ha="left"):
    spec = MODELS["mace_omat0"]
    line = ax.axvline if vertical else ax.axhline
    line(value, color=spec["color"], lw=0.9, ls=spec["linestyle"], zorder=1)
    if label and pos is not None:
        pad = " " if ha == "left" else ""
        ax.text(*pos, f"{pad}{text}{' ' if ha == 'right' else ''}", fontsize=5.5, color=INK["ink2"],
                va="center", ha=ha, zorder=4)


PROBLEMS: dict[str, list[str]] = {}


def save(fig, out: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    from .audit import check

    PROBLEMS[name] = check(fig)
    for p in PROBLEMS[name]:
        print(f"  {name}: LAYOUT {p}")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}: svg, pdf, png")


# ------------------------------------------------------------------- the data
def benchmark_panels(results: Path) -> list[tuple]:
    """(distance class, test, metric, {model: value}) for the six headline numbers."""
    val = _json(results / "eval/val90_heldout_all.json")["results"]
    val.update(_json(results / "eval/val90_heldout_omat0.json")["results"])
    mp = _json(results / "matpes/tables/matpes_tables.json")["metrics"]["f_mae_comp"]["domains"]
    mc = _json(results / "matcalc/matcalc_summary.json")["quantities"]
    st = _json(results / "stability/stability_summary.json")
    nb = _json(results / "nebdft2k/nebdft2k_summary.json")
    names = ("ours", "authors_envtok", "authors_direct", "mace_omat0", *SEEDS)
    val_key = {"ours": "ours_envtok", **{s: f"authors_{s}" for s in SEEDS}}

    def per(get):
        return {n: get(n) for n in names}

    return [
        ("ID", "OMat24 validation", "Force MAE (meV Å$^{-1}$)",
         per(lambda n: val[val_key.get(n, n)]["force_mae_meV_per_A"])),
        ("SHIFT", "MatPES bulk", "Force MAE, median (meV Å$^{-1}$)",
         per(lambda n: 1e3 * mp["bulk"]["estimators"]["median"][n])),
        ("OOD", "MatPES non-bulk", "Force MAE, median (meV Å$^{-1}$)",
         per(lambda n: 1e3 * mp["non_bulk"]["estimators"]["median"][n])),
        ("SHIFT", "MatCalc, MP binaries", "Shear modulus, median error (%)",
         per(lambda n: mc["gvrh"]["models"][n]["median"])),
        ("OOD", "MD stability, 80 NPT runs", "Runs physically diverged (%)",
         per(lambda n: 100 * st["ours@speed_of_authors_envtok" if n == "ours" else n]["n_diverged"]
             / st[n]["n_runs"])),
        ("OOD", "Li-ion migration, nebDFT2k", "Barrier MAE, no Hubbard U (meV)",
         per(lambda n: 1e3 * nb[n]["without_U"]["mae"])),
    ]


# -------------------------------------------------------------------- figures
@figure
def overview(results: Path, out: Path) -> None:
    """Where the four kinds of subset model stand on six benchmarks, against MACE-OMAT-0."""
    import matplotlib.pyplot as plt

    panels = benchmark_panels(results)
    with style():
        fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_IN, 3.8), sharey=True)
        for i, (ax, (kind, test, metric, vals)) in enumerate(zip(axes.flat, panels)):
            y = {m: k for k, m in enumerate(reversed(COMPARED))}
            seeds = [vals[s] for s in SEEDS]
            lo = min(min(seeds), *(vals[m] for m in COMPARED[:3]), vals["mace_omat0"])
            hi = max(max(seeds), *(vals[m] for m in COMPARED[:3]), vals["mace_omat0"])
            ax.set_xlim(lo - 0.13 * (hi - lo), hi + 0.13 * (hi - lo))
            left = vals["mace_omat0"] < (lo + hi) / 2
            reference(ax, vals["mace_omat0"], pos=(vals["mace_omat0"], len(COMPARED) - 0.42),
                      ha="left" if left else "right")
            ax.plot(seeds, [y["random"]] * len(seeds), "o", ms=3.0, color=color("random"),
                    mec="#8f8f8f", mew=0.4, zorder=2)
            for m in COMPARED:
                mark(ax, [np.mean(seeds) if m == "random" else vals[m]], [y[m]], m, ms=6.0)
            ax.set_yticks(list(y.values()), [MODELS[m]["label"].replace(" (4 seeds)", "")
                                              .replace(" (EnvTok)", "") for m in y])
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", visible=True)
            ax.set_ylim(-0.6, len(COMPARED) + 0.05)
            ax.set_title(test, loc="left", pad=4)
            ax.set_xlabel(metric)
            chip(ax, kind)
            letter(ax, "abcdef"[i], x=-0.02 if i % 3 else -0.42)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.11, hspace=0.78, wspace=0.16)
        save(fig, out, "overview")


@figure
def gap_id_ood(results: Path, out: Path) -> None:
    """How far each subset model stays behind MACE-OMAT-0, from in to out of distribution."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    sets = benchmark_panels(results)[:3]
    with style():
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.6))
        x = np.arange(3)

        def ratio(m):
            return np.array([p[3][m] / p[3]["mace_omat0"] for p in sets])

        seeds = np.array([ratio(s) for s in SEEDS])
        ax.fill_between(x, seeds.min(0), seeds.max(0), color=color("random"), alpha=0.35, lw=0,
                        zorder=1)
        reference(ax, 1.0, vertical=False, label=False)
        ax.text(-0.2, 1.012, "MACE-OMAT-0 (all of OMat24)", fontsize=5.5, color=INK["ink2"],
                va="bottom")
        for m in COMPARED:
            under = m == "random"  # context sits beneath the compared models
            yv = seeds.mean(0) if under else ratio(m)
            ax.plot(x, yv, color=line_color(m), lw=1.25, zorder=1.5 if under else 2)
            mark(ax, x, yv, m, ms=5.0 if under else 5.5, z=2.5 if under else 3)
        ax.set_xticks(x, ["OMat24 val.\nin distribution", "MatPES bulk\ndataset shift",
                          "MatPES non-bulk\nout of distribution"])
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0.9, 1.72)
        ax.set_ylabel("Force error relative to MACE-OMAT-0")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}×"))
        ax.legend(handles=[handle(m) for m in COMPARED], loc="upper left", ncol=2, fontsize=5.8,
                  columnspacing=1.0, handletextpad=0.4, borderaxespad=0.4)
        save(fig, out, "gap_id_ood")


@figure
def training(results: Path, out: Path) -> None:
    """Did training converge, and where did it stop?"""
    import matplotlib.pyplot as plt

    rows = _csv(results / "runs/envtok_8gpu/epochs.csv")
    ep = np.array([int(r["epoch"]) for r in rows])
    keep = ep >= 0
    ep = ep[keep]
    f = np.array([float(r["force_mae_meV_per_A"]) for r in rows])[keep]
    e = np.array([float(r["energy_mae_meV_per_atom"]) for r in rows])[keep]
    lr = np.array([float(r["lr"]) for r in rows])[keep]
    loss = np.array([float(r["loss"]) for r in rows])[keep]
    best = int(ep[np.argmin(loss)])
    ref = _json(results / "eval/val10_ours_epoch510_vs_authors.json")["results"]["authors_envtok"]
    with style():
        fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COLUMN_IN, 3.9), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1.15, 0.7]})
        ticks = ([100, 150, 200, 300], [25, 50, 100, 200])
        limits = ((92, 380), (19, 230))
        for ax, y, auth, lab, tk in ((axes[0], f, ref["force_mae_meV_per_A"], "Force MAE (meV Å$^{-1}$)", ticks[0]),
                                     (axes[1], e, ref["energy_mae_meV_per_atom"],
                                      "Energy MAE (meV atom$^{-1}$)", ticks[1])):
            ax.plot(ep, y, color=color("ours"), lw=1.0, zorder=2)
            ax.axhline(auth, color=color("authors_envtok"), lw=0.9, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(*limits[0 if ax is axes[0] else 1])
            ax.set_yticks(tk, [str(v) for v in tk])
            ax.minorticks_off()
            ax.set_ylabel(lab)
            ax.axvline(best, color=INK["grid"], lw=0.8, zorder=0)
            # below the reference line on the left, where our curve is still far above it;
            # the other annotation ends left of the best-epoch line
            ax.text(15, auth / 1.035, "authors' EnvTok model, same frames", fontsize=5.5,
                    color=INK["ink2"], ha="left", va="top")
            ax.text(best - 15, 0.93, f"this work: {y[ep == best][0]:.1f} at the best epoch",
                    transform=ax.get_xaxis_transform(), fontsize=5.5, color=INK["ink2"],
                    ha="right", va="top")
        axes[2].step(ep, lr, where="post", color=color("ours"), lw=1.0)
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Learning rate")
        axes[2].axvline(best, color=INK["grid"], lw=0.8, zorder=0)
        axes[2].text(best - 15, 1.15e-4, f"best epoch {best}, early stop {ep.max()}", fontsize=5.5,
                     color=INK["ink2"], ha="right", va="center")
        axes[2].set_xlabel("Epoch (777 optimiser steps of 1,024 structures each)")
        axes[2].set_xlim(-10, ep.max() + 10)
        for i, ax in enumerate(axes):
            letter(ax, "abc"[i], x=-0.2)
        axes[0].set_title("Validation: 65,775 OMat24 frames (10 % subset)", loc="left", pad=10)
        fig.align_ylabels(axes)
        fig.subplots_adjust(left=0.2, right=0.97, top=0.92, bottom=0.11, hspace=0.18)
        save(fig, out, "training")


@figure
def scaling(results: Path, out: Path) -> None:
    """How well does training scale from 1 to 8 GPUs spread over separate nodes?"""
    import matplotlib.pyplot as plt

    pts = _json(results / "benchmarks/pilot_8gpu_scattered.json")["scaling_steady_state"]
    g = np.array([p["gpus"] for p in pts])
    thr = np.array([p["structures_per_s"] for p in pts])
    with style():
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.5))
        ideal = thr[0] * g
        ax.plot(g, ideal, color=INK["muted"], lw=0.9, ls=(0, (4, 2.5)), zorder=1)
        ax.text(5.2, thr[0] * 5.2 * 1.04, "linear scaling", fontsize=5.5, color=INK["ink2"],
                rotation=0, ha="right", va="bottom")
        ax.plot(g, thr, color=color("ours"), lw=1.25, zorder=2)
        mark(ax, g, thr, "ours", ms=6.0)
        for p in pts:
            ax.text(p["gpus"] * 1.07, p["structures_per_s"] * 0.93, f"{100 * p['efficiency']:.0f} %",
                    fontsize=6, color=INK["ink"], ha="left", va="top")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(g, [f"{p['gpus']}\n{p['nodes']} node{'s' if p['nodes'] > 1 else ''}"
                          for p in pts])
        ax.set_yticks([1000, 2000, 4000, 8000, 16000], ["1k", "2k", "4k", "8k", "16k"])
        ax.minorticks_off()
        ax.set_xlim(0.8, 11)
        ax.set_ylim(900, 16000)
        ax.set_xlabel("A100 GPUs (each a separate one-GPU allocation)")
        ax.set_ylabel("Training throughput (structures s$^{-1}$)")
        save(fig, out, "scaling")


@figure
def matpes_tail(results: Path, out: Path) -> None:
    """How heavy is the tail of the force-error distribution, and which models blow up?"""
    import matplotlib.pyplot as plt

    rows = _csv(results / "matpes/tables/f_mae_comp_exceedance.csv")
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.6), sharey=True)
        for i, (ax, dom, title) in enumerate(zip(axes, ("bulk", "non_bulk"),
                                                 ("MatPES bulk (242,319 structures)",
                                                  "MatPES non-bulk (80,766 structures)"))):
            sub = [r for r in rows if r["domain"] == dom]
            x = np.array([float(r["threshold_meV_per_A"]) for r in sub])

            def y(m):
                v = 100 * np.array([float(r[m]) for r in sub])
                return np.where(v > 0, v, np.nan)

            for s in SEEDS:
                ax.plot(x, y(s), color=line_color("random"), lw=0.8, zorder=1)
            ax.plot(x, y("mace_omat0"), color=color("mace_omat0"), lw=0.9,
                    ls=MODELS["mace_omat0"]["linestyle"], zorder=2)
            for m in ("authors_direct", "authors_envtok", "ours"):
                ax.plot(x, y(m), color=color(m), lw=1.25, zorder=3)
                mark(ax, x[4::10], y(m)[4::10], m, ms=4.0, z=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(10, 1e4 if dom == "bulk" else 1e6)
            ax.set_ylim(5e-4, 150)
            ax.set_xlabel("Per-structure force MAE (meV Å$^{-1}$)")
            ax.set_title(title, loc="left", pad=4)
            letter(ax, "ab"[i], x=-0.02 if i else -0.17)
            chip(ax, "SHIFT" if dom == "bulk" else "OOD")
        axes[0].set_ylabel("Atoms in structures with a larger error (%)")
        axes[0].set_yticks([1e-3, 1e-2, 0.1, 1, 10, 100], ["0.001", "0.01", "0.1", "1", "10", "100"])
        axes[1].legend(handles=[handle("ours"), handle("authors_envtok"), handle("authors_direct"),
                                handle("random", label="Authors, random (each seed)"),
                                handle("mace_omat0")], loc="lower left", fontsize=5.8,
                       handletextpad=0.4, borderaxespad=0.5)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.17, wspace=0.08)
        save(fig, out, "matpes_tail")


@figure
def md_divergence(results: Path, out: Path) -> None:
    """When do the MD runs blow up as temperature and pressure rise?"""
    import matplotlib.pyplot as plt

    rows = _csv(results / "stability/stability_runs.csv")
    t = np.linspace(0, 10, 501)

    def curve(model):
        runs = [r for r in rows if r["model"] == model]
        div = np.array([10 * float(r["nfs"]) for r in runs if r["outcome"] == "diverged"])
        return np.array([100 * (div <= ti).sum() / len(runs) for ti in t]), len(div), len(runs)

    with style():
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.6))
        for s in SEEDS:
            ax.step(t, curve(s)[0], where="post", color=line_color("random"), lw=0.8, zorder=1)
        c, n, _ = curve("mace_omat0")
        ax.step(t, c, where="post", color=color("mace_omat0"), lw=0.9,
                ls=MODELS["mace_omat0"]["linestyle"], zorder=2)
        ends = {}
        for m, key in (("authors_direct", "authors_direct"), ("authors_envtok", "authors_envtok"),
                       ("ours", "ours@speed_of_authors_envtok")):
            c, n, total = curve(key)
            ax.step(t, c, where="post", color=color(m), lw=1.25, zorder=3)
            ends[m] = (c[-1], n, total)
        ax.text(9.85, ends["ours"][0] + 2.0, "This work: 0 of 80", fontsize=5.8, color=INK["ink"],
                ha="right", va="bottom")
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 52)
        ax.set_xticks([0, 2.5, 5, 7.5, 10], ["0 ps\n300 K\n0 GPa", "2.5\n975\n125", "5\n1,650\n250",
                                            "7.5\n2,325\n375", "10\n3,000\n500"])
        ax.set_xlabel("Simulation time, temperature and pressure (linear ramps)")
        ax.set_ylabel("Runs physically diverged (%)")
        ax.legend(handles=[handle("ours", label="This work (at the authors' step speed)", marker=False),
                           handle("authors_envtok", marker=False),
                           handle("authors_direct", marker=False),
                           handle("random", label="Authors, random (each seed)", marker=False),
                           handle("mace_omat0")],
                  loc="upper left", fontsize=5.6, handletextpad=0.4, borderaxespad=0.4)
        save(fig, out, "md_divergence")


@figure
def latent_distance(results: Path, out: Path) -> None:
    """Are atoms far from the training data in latent space the ones with large force errors?"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .style import DISTANCE_LINESTYLE

    sets = (("omat24_val", "OMat24 validation", "ID"), ("matpes_bulk", "MatPES bulk", "SHIFT"),
            ("matpes_non_bulk", "MatPES non-bulk", "OOD"))
    with style():
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 3.9), sharex=True)
        for col, model in enumerate(("ours", "authors_envtok")):
            d = _json(results / f"latent/latent_{model}.json")
            x = np.arange(1, 11)
            handles = []
            for key, name, kind in sets:
                s = d["sets"][key]
                share = np.array([100 * b["n"] / s["n_atoms"] for b in s["error_by_distance"]])
                err = np.array([1e3 * b["median_err"] for b in s["error_by_distance"]])
                ls = DISTANCE_LINESTYLE[kind]
                for ax, yv in ((axes[0, col], share), (axes[1, col], err)):
                    ax.plot(x, yv, color=color(model), lw=1.25, ls=ls, zorder=2)
                    mark(ax, x, yv, model, ms=4.0)
                handles.append(Line2D([], [], color=color(model), lw=1.25, ls=ls,
                                      label=f"{name} ({kind}), ρ = {s['spearman_distance_vs_error']:.2f}"))
            title = "This work" if model == "ours" else "Authors' EnvTok model"
            axes[0, col].set_title(f"{title} (PCA, {d['pca_dims']} components)", loc="left", pad=4)
            axes[0, col].set_ylim(0, 37)
            axes[1, col].set_yscale("log")
            axes[1, col].set_ylim(35, 1200)
            axes[1, col].set_yticks([50, 100, 200, 500, 1000], ["50", "100", "200", "500", "1,000"])
            axes[1, col].minorticks_off()
            axes[1, col].set_xticks(x)
            axes[1, col].set_xlabel("Distance to the 5 nearest training atoms\n"
                                    "(decile of all test atoms; 1 = nearest)")
            axes[1, col].legend(handles=handles, loc="upper left", fontsize=5.6, handlelength=2.4,
                                handletextpad=0.5, borderaxespad=0.5)
        axes[0, 0].set_ylabel("Share of the set's atoms (%)")
        axes[1, 0].set_ylabel("Median force error (meV Å$^{-1}$)")
        for i, ax in enumerate(axes.flat):
            letter(ax, "abcd"[i], x=-0.02 if i % 2 else -0.13)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.14, hspace=0.22, wspace=0.16)
        save(fig, out, "latent_distance")


@figure
def neb_parity(results: Path, out: Path) -> None:
    """Are the predicted Li-ion migration barriers biased against DFT?"""
    import matplotlib.pyplot as plt

    rows = [r for r in _csv(results / "nebdft2k/nebdft2k_hops.csv") if r["kept_ours"] == "True"]
    nb = _json(results / "nebdft2k/nebdft2k_summary.json")
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(ONE_HALF_COLUMN_IN, 2.75))
        for i, (ax, with_u, title) in enumerate(zip(axes, (False, True),
                                                    ("Hosts without Hubbard U", "Hosts with Hubbard U"))):
            sub = [r for r in rows if (r["has_specific_TM"] == "True") == with_u]
            dft = np.array([float(r["em_dft"]) for r in sub])
            pred = np.array([float(r["em_pred_ours"]) for r in sub])
            ax.plot([3e-3, 100], [3e-3, 100], color=INK["muted"], lw=0.8, zorder=1)
            ax.plot(dft, pred, "o", ms=2.2, color=color("ours"), alpha=0.45, mec="none", zorder=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # no barrier reaches 10 eV, so the top decade stays free for the statistics
            ax.set_xlim(3e-3, 100)
            ax.set_ylim(3e-3, 100)
            ax.set_aspect("equal")
            split = "with_U" if with_u else "without_U"
            ours, auth = nb["ours"][split], nb["authors_envtok"][split]
            ax.text(0.04, 0.96, f"n = {len(sub)}\nMAE {1e3 * ours['mae']:.0f} meV "
                                f"(authors' EnvTok {1e3 * auth['mae']:.0f})\n"
                                f"median signed error {1e3 * ours['median_signed']:+.0f} meV",
                    transform=ax.transAxes, fontsize=5.6, color=INK["ink"], va="top", ha="left")
            ax.set_xlabel("DFT barrier (eV)")
            ax.set_title(title, loc="left", pad=4)
            letter(ax, "ab"[i], x=-0.02 if i else -0.2)
        axes[0].set_ylabel("Predicted barrier (eV)")
        fig.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.16, wspace=0.28)
        save(fig, out, "neb_parity")


# ------------------------------------------------------------------------ CLI
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--only", nargs="*", default=None, choices=sorted(FIGURES))


def run(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    for name in args.only or FIGURES:
        FIGURES[name](Path(args.results), Path(args.out))
