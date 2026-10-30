"""Every figure of the repository, drawn from the files in ``results/`` only.

    envtok-mace figures --results results --out results/figures

Each figure answers one question and is written as SVG (editable text), PDF and
a 300 dpi PNG. Colours, markers, sizes and fonts come from ``plots.style``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..analysis.results_io import MODELS as NAMES
from ..analysis.results_io import SEEDS, SPEED_MATCHED, load, read_csv, read_json
from .style import (
    DOUBLE_COLUMN_IN,
    EDGE_PT,
    INK,
    MODELS,
    ONE_HALF_COLUMN_IN,
    SINGLE_COLUMN_IN,
    color,
    line_color,
    style,
)

COMPARED = ("ours", "authors_envtok", "authors_direct", "random")
FIGURES: dict[str, callable] = {}


def figure(fn):
    FIGURES[fn.__name__] = fn
    return fn


# --------------------------------------------------------------------- helpers
def mark(ax, x, y, model, ms=5.5, z=3, **kw):
    spec = MODELS[model]
    ax.plot(
        x,
        y,
        spec["marker"],
        color=spec["color"],
        ms=ms,
        mec=spec["edge"],
        mew=EDGE_PT,
        zorder=z,
        linestyle="none",
        **kw,
    )


def handle(model, line=True, label=None, marker=True):
    from matplotlib.lines import Line2D

    spec = MODELS[model]
    if spec["marker"] is None or not marker:
        if spec["marker"] is not None:
            return Line2D([], [], color=line_color(model), lw=1.25, label=label or spec["label"])
        return Line2D([], [], color=spec["color"], lw=0.9, ls=spec["linestyle"], label=label or spec["label"])
    return Line2D(
        [],
        [],
        color=line_color(model) if line else "none",
        lw=1.25,
        marker=spec["marker"],
        ms=4.5,
        mfc=spec["color"],
        mec=spec["edge"],
        mew=EDGE_PT,
        label=label or spec["label"],
    )


def chip(ax, text):
    ax.text(
        1.0,
        1.03,
        text,
        transform=ax.transAxes,
        fontsize=5.5,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=INK["ink2"],
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc=INK["chip"], ec="none"),
    )


def letter(ax, text, x=-0.02):
    ax.text(
        x,
        1.03,
        text,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=INK["ink"],
    )


def reference(ax, value, label=True, vertical=True, text="MACE-OMAT-0", pos=None, ha="left"):
    spec = MODELS["mace_omat0"]
    line = ax.axvline if vertical else ax.axhline
    line(value, color=spec["color"], lw=0.9, ls=spec["linestyle"], zorder=1)
    if label and pos is not None:
        pad = " " if ha == "left" else ""
        ax.text(
            *pos,
            f"{pad}{text}{' ' if ha == 'right' else ''}",
            fontsize=5.5,
            color=INK["ink2"],
            va="center",
            ha=ha,
            zorder=4,
        )


FONT_STACK = "'Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans', sans-serif"


def _font_fallbacks(path: Path) -> None:
    """Give every font declaration in an SVG the full sans-serif stack.

    With editable SVG text matplotlib names only the font it used ('Arial', or
    'DejaVu Sans' for mathtext), and a browser without that font falls back to
    its default, often a serif face. The stack keeps every viewer on sans-serif.
    """
    import re

    text = path.read_text()
    text = re.sub(
        r"font-family: ?'[^;\"]*?'(?=[;\"])",
        lambda m: m.group(0) if "sans-serif" in m.group(0) else f"font-family: {FONT_STACK}",
        text,
    )
    path.write_text(text)


PROBLEMS: dict[str, list[str]] = {}


def save(fig, out: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    from .audit import check

    PROBLEMS[name] = check(fig)
    for p in PROBLEMS[name]:
        print(f"  {name}: LAYOUT {p}")
    out.mkdir(parents=True, exist_ok=True)
    # no creation dates in the files, so a figure redrawn from the same data is byte-identical
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    _font_fallbacks(out / f"{name}.svg")
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}: svg, pdf, png")


# ------------------------------------------------------------------- the data
def benchmark_panels(results: Path) -> list[tuple]:
    """(distance class, test, metric, {model: value}) for the six headline numbers."""
    r = load(results)
    val, mc, st, nb = r["val"], r["matcalc"], r["stability"], r["neb"]
    mp = r["matpes"]["f_mae_comp"]["domains"]

    def per(get):
        return {n: get(n) for n in NAMES}

    return [
        (
            "ID",
            "OMat24 validation",
            "Force MAE (meV Å$^{-1}$)",
            per(lambda n: val[n]["force_mae_meV_per_A"]),
        ),
        (
            "SHIFT",
            "MatPES bulk",
            "Force MAE, median (meV Å$^{-1}$)",
            per(lambda n: 1e3 * mp["bulk"]["estimators"]["median"][n]),
        ),
        (
            "OOD",
            "MatPES non-bulk",
            "Force MAE, median (meV Å$^{-1}$)",
            per(lambda n: 1e3 * mp["non_bulk"]["estimators"]["median"][n]),
        ),
        (
            "SHIFT",
            "MatCalc, MP binaries",
            "Shear modulus, median error (%)",
            per(lambda n: mc["gvrh"]["models"][n]["median"]),
        ),
        (
            "OOD",
            "MD stability, 80 NPT runs",
            "Runs physically diverged (%)",
            per(lambda n: 100 * st[SPEED_MATCHED if n == "ours" else n]["n_diverged"] / st[n]["n_runs"]),
        ),
        (
            "OOD",
            "Li-ion migration, nebDFT2k",
            "Barrier MAE, no Hubbard U (meV)",
            per(lambda n: 1e3 * nb[n]["without_U"]["mae"]),
        ),
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
            reference(
                ax,
                vals["mace_omat0"],
                pos=(vals["mace_omat0"], len(COMPARED) - 0.42),
                ha="left" if left else "right",
            )
            ax.plot(
                seeds,
                [y["random"]] * len(seeds),
                "o",
                ms=3.0,
                color=color("random"),
                mec="#8f8f8f",
                mew=0.4,
                zorder=2,
            )
            for m in COMPARED:
                mark(ax, [np.mean(seeds) if m == "random" else vals[m]], [y[m]], m, ms=6.0)
            ax.set_yticks(
                list(y.values()),
                [MODELS[m]["label"].replace(" (4 seeds)", "").replace(" (EnvTok)", "") for m in y],
            )
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
        ax.fill_between(x, seeds.min(0), seeds.max(0), color=color("random"), alpha=0.35, lw=0, zorder=1)
        reference(ax, 1.0, vertical=False, label=False)
        ax.text(-0.2, 1.012, "MACE-OMAT-0 (all of OMat24)", fontsize=5.5, color=INK["ink2"], va="bottom")
        for m in COMPARED:
            under = m == "random"  # context sits beneath the compared models
            yv = seeds.mean(0) if under else ratio(m)
            ax.plot(x, yv, color=line_color(m), lw=1.25, zorder=1.5 if under else 2)
            mark(ax, x, yv, m, ms=5.0 if under else 5.5, z=2.5 if under else 3)
        ax.set_xticks(
            x,
            [
                "OMat24 val.\nin distribution",
                "MatPES bulk\ndataset shift",
                "MatPES non-bulk\nout of distribution",
            ],
        )
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0.9, 1.72)
        ax.set_ylabel("Force error relative to MACE-OMAT-0")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}×"))
        ax.legend(
            handles=[handle(m) for m in COMPARED],
            loc="upper left",
            ncol=2,
            fontsize=5.8,
            columnspacing=1.0,
            handletextpad=0.4,
            borderaxespad=0.4,
        )
        save(fig, out, "gap_id_ood")


def _log_ticks(ax, values) -> None:
    """1-2-3-5 ticks spanning ``values`` on a log axis, labelled as plain numbers."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    ticks = [m * 10.0**d for d in decades for m in (1, 2, 3, 5) if lo / 1.15 <= m * 10.0**d <= hi * 1.15]
    ax.set_ylim(lo / 1.15, hi * 1.15)
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average; near the ends it averages over the points that exist."""
    kernel = np.ones(window)
    return np.convolve(y, kernel, mode="same") / np.convolve(np.ones_like(y), kernel, mode="same")


TRAIN_LOSS_WINDOW = 9  # epochs; each epoch's training loss is the mean of only ~15 logged steps

# how each data set is drawn in the learning curves; every line is this work's model
# Validation and test nearly coincide, so the test markers are open and the validation
# line is drawn over them.
LEARNING_SETS = {
    "train": {"label": "Training set", "ls": (0, (4, 2)), "marker": "v", "mfc": None, "z": 2},
    "val": {
        "label": "Validation set (selects the best epoch)",
        "ls": "-",
        "marker": None,
        "mfc": None,
        "z": 3,
    },
    "test": {"label": "Held-out test set", "ls": (0, (1.2, 1.6)), "marker": "o", "mfc": "white", "z": 2},
}
LEARNING_MARKER = 2.6


@figure
def training(results: Path, out: Path) -> None:
    """Did training converge, did it overfit, and where did it stop?

    Validation is scored after every epoch. The training loss is the mean of the step
    losses logged during an epoch (``train_loss.csv``, about 15 steps per epoch), drawn
    as a centred moving average over ``TRAIN_LOSS_WINDOW`` epochs. The training- and
    test-set errors come from the checkpoints archived every 25 epochs
    (``checkpoints.csv``). Each is drawn when its table is in ``results/``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    run = results / "runs/envtok_8gpu"
    rows = [r for r in read_csv(run / "epochs.csv") if int(r["epoch"]) >= 0]
    ep = np.array([int(r["epoch"]) for r in rows])
    col = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "epoch"}
    best = int(ep[np.argmin(col["loss"])])
    ref = read_json(results / "eval/val10_authors_envtok.json")["results"]["authors_envtok"]
    train_loss = read_csv(run / "train_loss.csv") if (run / "train_loss.csv").exists() else []
    ckpt = read_csv(run / "checkpoints.csv") if (run / "checkpoints.csv").exists() else []
    navy = color("ours")

    def curve(ax, x, y, which, every_epoch=False):
        """Markers only on the archived checkpoints; a value every epoch is a plain line."""
        s = LEARNING_SETS[which]
        marker = None if every_epoch else s["marker"]
        ax.plot(
            x, y, color=navy, lw=1.0, ls=s["ls"], marker=marker, ms=LEARNING_MARKER,
            mfc=s["mfc"] or navy, mew=0.8, zorder=s["z"],
        )  # fmt: skip

    with style():
        # one row of the four curves, the learning rate as a strip underneath, the key on top
        fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 3.1))
        grid = fig.add_gridspec(
            2,
            4,
            height_ratios=[1, 0.3],
            left=0.055,
            right=0.99,
            top=0.8,
            bottom=0.13,
            hspace=0.6,
            wspace=0.34,
        )
        axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
        lr_ax = fig.add_subplot(grid[1, :])
        panels = (
            (axes[0], "loss", "Loss (weighted Huber)", None),
            (axes[1], "force_mae_meV_per_A", "Force MAE (meV Å$^{-1}$)", "force_mae_meV_per_A"),
            (axes[2], "energy_mae_meV_per_atom", "Energy MAE (meV atom$^{-1}$)", "energy_mae_meV_per_atom"),
            (axes[3], "stress_mae_meV_per_A3", "Stress MAE (meV Å$^{-3}$)", "stress_mae_meV_per_A3"),
        )
        for ax, key, title, ref_key in panels:
            shown = [col[key]]
            curve(ax, ep, col[key], "val", every_epoch=True)
            if key == "loss" and train_loss:
                y = _moving_average(np.array([float(r["loss_mean"]) for r in train_loss]), TRAIN_LOSS_WINDOW)
                curve(ax, [int(r["epoch"]) for r in train_loss], y, "train", every_epoch=True)
                shown.append(y)
            for which in ("train", "test"):
                pts = [r for r in ckpt if r["set"] == which]
                if pts:  # eval scores the same loss, so the checkpoints go on the loss panel too
                    y = np.array([float(r[key]) for r in pts])
                    curve(ax, [int(r["epoch"]) for r in pts], y, which)
                    shown.append(y)
            if ref_key:
                ax.axhline(ref[ref_key], color=color("authors_envtok"), lw=0.9, zorder=1)
                shown.append([ref[ref_key]])
            ax.axvline(best, color=INK["grid"], lw=0.8, zorder=0)
            ax.set_yscale("log")
            _log_ticks(ax, np.concatenate([np.ravel(v) for v in shown]))
            ax.set_title(title, loc="left", pad=4)
            ax.set_xlim(-10, ep.max() + 10)
            ax.set_xticks([0, 200, 400, 600])
        lr_ax.step(ep, col["lr"], where="post", color=navy, lw=1.0)
        lr_ax.axvline(best, color=INK["grid"], lw=0.8, zorder=0)
        lr_ax.set_yscale("log")
        lr_ax.set_xlim(-10, ep.max() + 10)
        lr_ax.set_title("Learning rate", loc="left", pad=3)
        lr_ax.set_xlabel("Epoch (777 optimiser steps of 1,024 structures each)")
        for i, ax in enumerate([*axes, lr_ax]):
            letter(ax, "abcde"[i], x=-0.1 if i < 4 else -0.045)

        drawn = ["val"] + (["train"] if train_loss or ckpt else []) + (["test"] if ckpt else [])
        # markers stand for the archived checkpoints, so they appear in the key only with them
        handles = [
            Line2D(
                [],
                [],
                color=navy,
                lw=1.0,
                ls=s["ls"],
                marker=s["marker"] if ckpt else None,
                ms=LEARNING_MARKER,
                mfc=s["mfc"] or navy,
                mew=0.8,
                label=s["label"],
            )
            for k, s in LEARNING_SETS.items()
            if k in drawn
        ]
        handles.append(
            Line2D(
                [], [], color=color("authors_envtok"), lw=0.9, label="Authors' EnvTok model, validation set"
            )
        )
        handles.append(
            Line2D([], [], color=INK["grid"], lw=0.8, label=f"Best epoch {best} (early stop {ep.max()})")
        )
        fig.legend(
            handles=handles,
            loc="lower left",  # bottom edge fixed above the titles; more entries grow upwards
            bbox_to_anchor=(0.045, 0.88),
            ncol=3,
            fontsize=6,
            handlelength=2.6,
            columnspacing=1.6,
        )
        save(fig, out, "training")


@figure
def scaling(results: Path, out: Path) -> None:
    """How well does training scale from 1 to 8 GPUs spread over separate nodes?"""
    import matplotlib.pyplot as plt

    pts = read_json(results / "benchmarks/pilot_8gpu_scattered.json")["scaling_steady_state"]
    g = np.array([p["gpus"] for p in pts])
    thr = np.array([p["structures_per_s"] for p in pts])
    with style():
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.5))
        ideal = thr[0] * g
        ax.plot(g, ideal, color=INK["muted"], lw=0.9, ls=(0, (4, 2.5)), zorder=1)
        ax.text(
            5.2,
            thr[0] * 5.2 * 1.04,
            "linear scaling",
            fontsize=5.5,
            color=INK["ink2"],
            rotation=0,
            ha="right",
            va="bottom",
        )
        ax.plot(g, thr, color=color("ours"), lw=1.25, zorder=2)
        mark(ax, g, thr, "ours", ms=6.0)
        for p in pts:
            ax.text(
                p["gpus"] * 1.07,
                p["structures_per_s"] * 0.93,
                f"{100 * p['efficiency']:.0f} %",
                fontsize=6,
                color=INK["ink"],
                ha="left",
                va="top",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(g, [f"{p['gpus']}\n{p['nodes']} node{'s' if p['nodes'] > 1 else ''}" for p in pts])
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

    rows = read_csv(results / "matpes/tables/f_mae_comp_exceedance.csv")
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.6), sharey=True)
        for i, (ax, dom, title) in enumerate(
            zip(
                axes,
                ("bulk", "non_bulk"),
                ("MatPES bulk (242,319 structures)", "MatPES non-bulk (80,766 structures)"),
            )
        ):
            sub = [r for r in rows if r["domain"] == dom]
            x = np.array([float(r["threshold_meV_per_A"]) for r in sub])

            def y(m, sub=sub):
                v = 100 * np.array([float(r[m]) for r in sub])
                return np.where(v > 0, v, np.nan)

            for s in SEEDS:
                ax.plot(x, y(s), color=line_color("random"), lw=0.8, zorder=1)
            ax.plot(
                x,
                y("mace_omat0"),
                color=color("mace_omat0"),
                lw=0.9,
                ls=MODELS["mace_omat0"]["linestyle"],
                zorder=2,
            )
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
        axes[1].legend(
            handles=[
                handle("ours"),
                handle("authors_envtok"),
                handle("authors_direct"),
                handle("random", label="Authors, random (each seed)"),
                handle("mace_omat0"),
            ],
            loc="lower left",
            fontsize=5.8,
            handletextpad=0.4,
            borderaxespad=0.5,
        )
        fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.17, wspace=0.08)
        save(fig, out, "matpes_tail")


@figure
def md_divergence(results: Path, out: Path) -> None:
    """When do the MD runs blow up as temperature and pressure rise?"""
    import matplotlib.pyplot as plt

    rows = read_csv(results / "stability/stability_runs.csv")
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
        ax.step(
            t,
            c,
            where="post",
            color=color("mace_omat0"),
            lw=0.9,
            ls=MODELS["mace_omat0"]["linestyle"],
            zorder=2,
        )
        ends = {}
        for m, key in (
            ("authors_direct", "authors_direct"),
            ("authors_envtok", "authors_envtok"),
            ("ours", SPEED_MATCHED),
        ):
            c, n, total = curve(key)
            ax.step(t, c, where="post", color=color(m), lw=1.25, zorder=3)
            ends[m] = (c[-1], n, total)
        ax.text(
            9.85,
            ends["ours"][0] + 2.0,
            "This work: 0 of 80",
            fontsize=5.8,
            color=INK["ink"],
            ha="right",
            va="bottom",
        )
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 52)
        ax.set_xticks(
            [0, 2.5, 5, 7.5, 10],
            ["0 ps\n300 K\n0 GPa", "2.5\n975\n125", "5\n1,650\n250", "7.5\n2,325\n375", "10\n3,000\n500"],
        )
        ax.set_xlabel("Simulation time, temperature and pressure (linear ramps)")
        ax.set_ylabel("Runs physically diverged (%)")
        ax.legend(
            handles=[
                handle("ours", label="This work (at the authors' step speed)", marker=False),
                handle("authors_envtok", marker=False),
                handle("authors_direct", marker=False),
                handle("random", label="Authors, random (each seed)", marker=False),
                handle("mace_omat0"),
            ],
            loc="upper left",
            fontsize=5.6,
            handletextpad=0.4,
            borderaxespad=0.4,
        )
        save(fig, out, "md_divergence")


@figure
def latent_distance(results: Path, out: Path) -> None:
    """Are atoms far from the training data in latent space the ones with large force errors?"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .style import DISTANCE_LINESTYLE

    sets = (
        ("omat24_val", "OMat24 validation", "ID"),
        ("matpes_bulk", "MatPES bulk", "SHIFT"),
        ("matpes_non_bulk", "MatPES non-bulk", "OOD"),
    )
    with style():
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 3.9), sharex=True)
        for col, model in enumerate(("ours", "authors_envtok")):
            d = read_json(results / f"latent/latent_{model}.json")
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
                handles.append(
                    Line2D(
                        [],
                        [],
                        color=color(model),
                        lw=1.25,
                        ls=ls,
                        label=f"{name} ({kind}), ρ = {s['spearman_distance_vs_error']:.2f}",
                    )
                )
            title = "This work" if model == "ours" else "Authors' EnvTok model"
            axes[0, col].set_title(f"{title} (PCA, {d['pca_dims']} components)", loc="left", pad=4)
            axes[0, col].set_ylim(0, 37)
            axes[1, col].set_yscale("log")
            axes[1, col].set_ylim(35, 1200)
            axes[1, col].set_yticks([50, 100, 200, 500, 1000], ["50", "100", "200", "500", "1,000"])
            axes[1, col].minorticks_off()
            axes[1, col].set_xticks(x)
            axes[1, col].set_xlabel(
                "Distance to the 5 nearest training atoms\n(decile of all test atoms; 1 = nearest)"
            )
            axes[1, col].legend(
                handles=handles,
                loc="upper left",
                fontsize=5.6,
                handlelength=2.4,
                handletextpad=0.5,
                borderaxespad=0.5,
            )
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

    rows = [r for r in read_csv(results / "nebdft2k/nebdft2k_hops.csv") if r["kept_ours"] == "True"]
    nb = read_json(results / "nebdft2k/nebdft2k_summary.json")
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(ONE_HALF_COLUMN_IN, 2.75))
        for i, (ax, with_u, title) in enumerate(
            zip(axes, (False, True), ("Hosts without Hubbard U", "Hosts with Hubbard U"))
        ):
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
            ax.text(
                0.04,
                0.96,
                f"n = {len(sub)}\nMAE {1e3 * ours['mae']:.0f} meV "
                f"(authors' EnvTok {1e3 * auth['mae']:.0f})\n"
                f"median signed error {1e3 * ours['median_signed']:+.0f} meV",
                transform=ax.transAxes,
                fontsize=5.6,
                color=INK["ink"],
                va="top",
                ha="left",
            )
            ax.set_xlabel("DFT barrier (eV)")
            ax.set_title(title, loc="left", pad=4)
            letter(ax, "ab"[i], x=-0.02 if i else -0.2)
        axes[0].set_ylabel("Predicted barrier (eV)")
        fig.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.16, wspace=0.28)
        save(fig, out, "neb_parity")


# ------------------------------------------------------ accuracy against DFT
# figure -> folder in results/accuracy, domain, title, distance from the training data
ACCURACY_SETS = {
    "parity": ("omat24_val90", "all", "OMat24 held-out", "ID"),
    "parity_matpes_bulk": ("matpes", "bulk", "MatPES bulk", "SHIFT"),
    "parity_matpes_non_bulk": ("matpes", "non_bulk", "MatPES non-bulk", "OOD"),
}
PARITY_ROWS = (
    # quantity, axis label, summary keys of the MAE and RMSE, their unit
    (
        "energy",
        "energy (eV atom$^{-1}$)",
        "energy_mae_meV_per_atom",
        "energy_rmse_meV_per_atom",
        "meV atom$^{-1}$",
    ),
    (
        "forces",
        "force component (eV Å$^{-1}$)",
        "force_mae_meV_per_A",
        "force_rmse_meV_per_A",
        "meV Å$^{-1}$",
    ),
    ("stress", "stress component (GPa)", "stress_mae_voigt6_meV_per_A3", None, "meV Å$^{-3}$"),
)
FORCE_TICKS = (-100, -10, -1, 0, 1, 10, 100)  # eV/A
GPA_PER_MEV_PER_A3 = 0.16021766208
ACCURACY_MODELS = ("ours", "authors_envtok", "mace_omat0")


def _parity(results: Path, out: Path, name: str) -> None:
    """Predicted against DFT energy, forces and stress, one column per model."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.ticker import MaxNLocator

    folder, domain, title, distance = ACCURACY_SETS[name]
    par = read_json(results / "accuracy" / folder / "parity.json")
    summ = read_json(results / "accuracy" / folder / "summary.json")
    models = [m for m in ACCURACY_MODELS if m in par["counts"]]
    cmap = LinearSegmentedColormap.from_list("density", ["#c9d3e0", color("ours")])
    norm = LogNorm(1e-6, 1e-1)
    with style():
        fig, axes = plt.subplots(3, len(models), figsize=(DOUBLE_COLUMN_IN, 7.5), squeeze=False)
        for col, model in enumerate(models):
            stats = summ["models"][model]["domains"][domain]
            for row, (q, axis, mae, rmse, unit) in enumerate(PARITY_ROWS):
                ax = axes[row, col]
                edges = np.asarray(par["edges"][q])
                c = par["counts"][model][domain][q]
                h = np.zeros((len(edges) - 1,) * 2)
                if c["ijc"]:
                    i, j, k = np.asarray(c["ijc"]).T
                    h[i, j] = k
                share = np.where(h > 0, h / c["n"], np.nan)
                mesh = ax.pcolormesh(edges, edges, share.T, cmap=cmap, norm=norm, rasterized=True, zorder=1)
                lo, hi = edges[0], edges[-1]
                ax.plot([lo, hi], [lo, hi], color=INK["muted"], lw=0.6, zorder=2)
                if q == "forces":
                    ticks = [t for t in FORCE_TICKS if lo <= t <= hi]
                    labels = [f"{t:g}".replace("-", "−") for t in ticks]
                    for scale, set_ticks in ((ax.set_xscale, ax.set_xticks), (ax.set_yscale, ax.set_yticks)):
                        scale("asinh", linear_width=0.1)
                        set_ticks(ticks, labels)
                    ax.minorticks_off()
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
                    ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal")
                text = f"MAE {stats[mae]:,.1f}"
                if rmse:
                    text += f", RMSE {stats[rmse]:,.1f} {unit}"
                else:
                    text += f" {unit} ({stats[mae] * GPA_PER_MEV_PER_A3:.2f} GPa)"
                text += f"\n{100 * c['n_outside'] / c['n']:.2g}% of the points off the axes"
                ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=5.6, va="bottom", ha="left")
                ax.set_xlabel(f"DFT {axis}")
                if col == 0:
                    ax.set_ylabel(f"Predicted {axis}")
                if row == 0:
                    ax.set_title(MODELS[model]["label"], loc="left", pad=20)
                letter(ax, "abcdefghi"[row * len(models) + col], x=-0.02 if col else -0.24)
        n = summ["models"][models[0]]["domains"][domain]["n_structures"]
        fig.text(0.01, 0.995, f"{title}, {n:,} structures", fontsize=7.5, va="top", ha="left")
        fig.text(
            0.89,
            0.995,
            distance,
            fontsize=5.5,
            fontweight="bold",
            va="top",
            ha="right",
            color=INK["ink2"],
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc=INK["chip"], ec="none"),
        )
        fig.subplots_adjust(left=0.1, right=0.89, top=0.91, bottom=0.065, hspace=0.46, wspace=0.36)
        cax = fig.add_axes([0.915, 0.38, 0.014, 0.3])
        bar = fig.colorbar(mesh, cax=cax)
        bar.set_label("Share of the points in a bin")
        bar.outline.set_linewidth(0.6)
        save(fig, out, name)


@figure
def parity(results: Path, out: Path) -> None:
    """How close are the predicted energies, forces and stresses to DFT on held-out OMat24?"""
    _parity(results, out, "parity")


@figure
def parity_matpes_bulk(results: Path, out: Path) -> None:
    """The same on MatPES bulk structures, a different DFT setting (dataset shift)."""
    _parity(results, out, "parity_matpes_bulk")


@figure
def parity_matpes_non_bulk(results: Path, out: Path) -> None:
    """The same on MatPES surfaces, molecules and clusters (out of distribution)."""
    _parity(results, out, "parity_matpes_non_bulk")


MIN_ELEMENT_ATOMS = 1000  # the rarest elements have 6 to 66 atoms; the next one has 62,451


def _cell_text_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK["ink"]


@figure
def element_errors(results: Path, out: Path) -> None:
    """Which elements carry the largest force errors, and where does ours differ from the authors' model?"""
    import matplotlib.pyplot as plt
    from ase.data import chemical_symbols
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle

    from ..data.constants import Z_TABLE
    from .ptable import LANTHANIDE_ROW, LAST_Z, position

    rows = [
        r
        for r in read_csv(results / "accuracy/omat24_val90/per_element.csv")
        if r["domain"] == "all" and int(r["n_atoms"]) >= MIN_ELEMENT_ATOMS
    ]
    mae = {
        m: {int(r["Z"]): float(r["force_mae_meV_per_A"]) for r in rows if r["model"] == m}
        for m in ("ours", "authors_envtok")
    }
    ratio = {z: v / mae["authors_envtok"][z] for z, v in mae["ours"].items()}
    panels = (
        (
            "This work, force MAE per element",
            "meV Å$^{-1}$",
            mae["ours"],
            LogNorm(min(mae["ours"].values()), max(mae["ours"].values())),
            LinearSegmentedColormap.from_list("mae", ["#eef1f5", "#6a8cb3", color("ours")]),
            "{:.0f}",
            (30, 50, 100, 200, 300, 500, 1000),
        ),
        (
            "This work against the authors' EnvTok model",
            "ratio of the force MAEs",
            ratio,
            LogNorm(0.8, 1.25),  # fixed: the ratios are within 5%, and the colour should not inflate them
            LinearSegmentedColormap.from_list("ratio", [color("ours"), "#ffffff", color("authors_direct")]),
            "{:.2f}",
            (0.8, 0.9, 1, 1.1, 1.25),
        ),
    )
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.05))
        for i, (ax, (title, unit, values, norm, cmap, fmt, bar_ticks)) in enumerate(zip(axes, panels)):
            for z in range(1, LAST_Z + 1):
                r, c = position(z)
                y = -(r - 0.6 if r == LANTHANIDE_ROW else r)
                v = values.get(z)
                if v is not None:
                    fc, ec, ink = cmap(norm(v)), "none", _cell_text_color(cmap(norm(v)))
                elif z in Z_TABLE:  # an element of the model without atoms in the held-out set
                    fc, ec, ink = INK["chip"], "none", INK["muted"]
                else:
                    fc, ec, ink = "none", INK["grid"], INK["muted"]
                ax.add_patch(Rectangle((c - 0.46, y - 0.46), 0.92, 0.92, fc=fc, ec=ec, lw=0.5))
                ax.text(c, y + 0.2, chemical_symbols[z], fontsize=5.2, color=ink, ha="center", va="center")
                if v is not None:
                    ax.text(c, y - 0.2, fmt.format(v), fontsize=5, color=ink, ha="center", va="center")
            ax.set_xlim(-0.55, 17.55)
            ax.set_ylim(-6.95, 0.55)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            ax.set_title(title, loc="left", pad=3)
            letter(ax, "ab"[i], x=-0.01)
            # the colour bar sits in the gap above the transition metals
            cax = ax.inset_axes([2.9, -1.25, 7.9, 0.3], transform=ax.transData)
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
            bar.outline.set_linewidth(0.5)
            bar.ax.tick_params(labelsize=5.2, length=1.8, width=0.5, pad=1.2)
            bar.ax.minorticks_off()
            ticks = [t for t in bar_ticks if norm.vmin <= t <= norm.vmax]
            bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
            bar.ax.set_title(unit, fontsize=5.4, pad=2, color=INK["ink2"])
        fig.text(
            0.5,
            0.0,
            f"Grey: fewer than {MIN_ELEMENT_ATOMS:,} atoms in the held-out set. "
            "Outlined: outside the model's 68 elements.",
            fontsize=5.4,
            color=INK["ink2"],
            ha="center",
            va="bottom",
        )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.04, wspace=0.04)
        save(fig, out, "element_errors")


@figure
def error_by_source(results: Path, out: Path) -> None:
    """Which kinds of OMat24 structures are hardest, and how many of them did training see?"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    folder = results / "accuracy/omat24_val90"
    rows = [r for r in read_csv(folder / "per_source.csv") if r["domain"] == "all"]
    train = {r["source"]: int(r["n_atoms"]) for r in read_csv(folder / "training_sources.csv")}
    order = (
        "rattled-relax",
        "rattled-300-subsampled",
        "rattled-300",
        "rattled-500-subsampled",
        "rattled-500",
        "rattled-1000-subsampled",
        "rattled-1000",
        "aimd-from-PBE-1000-nvt",
        "aimd-from-PBE-1000-npt",
        "aimd-from-PBE-3000-nvt",
        "aimd-from-PBE-3000-npt",
    )
    sources = [s for s in order if any(r["source"] == s for r in rows)]
    y = -np.arange(len(sources), dtype=float)

    def per(model, key):
        v = {r["source"]: float(r[key]) for r in rows if r["model"] == model}
        return np.array([v[s] for s in sources])

    held = {r["source"]: int(r["n_atoms"]) for r in rows if r["model"] == "ours"}
    shares = {
        "training": 100 * np.array([train.get(s, 0) for s in sources]) / sum(train.values()),
        "held-out": 100 * np.array([held[s] for s in sources]) / sum(held.values()),
    }
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.9), sharey=True)
        for i, (ax, key, label) in enumerate(
            zip(
                axes,
                ("force_mae_meV_per_A", "energy_mae_meV_per_atom"),
                ("Force MAE (meV Å$^{-1}$)", "Energy MAE (meV atom$^{-1}$)"),
            )
        ):
            for yi in y:
                ax.axhline(yi, color=INK["grid"], lw=0.4, zorder=0)
            ax.plot(per("mace_omat0", key), y, "|", ms=7, mew=1.1, color=color("mace_omat0"), zorder=2)
            for m in ("authors_envtok", "ours"):
                mark(ax, per(m, key), y, m, ms=4.5)
            ax.set_xlabel(label)
            ax.set_xlim(0, None)
            letter(ax, "abc"[i], x=-0.02 if i else -0.75)
        ax = axes[2]
        for v, dy, fc in zip(shares.values(), (0.19, -0.19), (INK["ink2"], INK["surface"])):
            ax.barh(y + dy, v, height=0.36, color=fc, edgecolor=INK["ink2"], lw=0.6, zorder=2)
        ax.set_xlabel("Share of the set's atoms (%)")
        ax.set_xlim(0, None)
        letter(ax, "c")
        axes[0].set_yticks(y, sources)
        axes[0].set_ylim(y[-1] - 0.6, 0.6)
        axes[0].legend(
            handles=[
                handle("ours", line=False, label="This work"),
                handle("authors_envtok", line=False),
                Line2D(
                    [],
                    [],
                    color=color("mace_omat0"),
                    marker="|",
                    ms=7,
                    mew=1.1,
                    ls="none",
                    label="MACE-OMAT-0",
                ),
            ],
            loc="lower left",
            bbox_to_anchor=(0, 1.09),
            ncol=3,
            fontsize=5.6,
            handletextpad=0.3,
            columnspacing=1.0,
            borderaxespad=0.2,
            borderpad=0,
        )
        ax.legend(
            handles=[
                Patch(fc=INK["ink2"], ec=INK["ink2"], lw=0.6, label="Training set"),
                Patch(fc=INK["surface"], ec=INK["ink2"], lw=0.6, label="Held-out set"),
            ],
            loc="lower left",
            bbox_to_anchor=(0, 1.09),
            ncol=2,
            fontsize=5.6,
            handlelength=1.2,
            columnspacing=1.0,
            borderaxespad=0.2,
            borderpad=0,
        )
        fig.subplots_adjust(left=0.2, right=0.99, top=0.87, bottom=0.15, wspace=0.12)
        save(fig, out, "error_by_source")


@figure
def error_vs_force(results: Path, out: Path) -> None:
    """Does the force error grow with the size of the DFT force, in and out of distribution?"""
    import matplotlib.pyplot as plt

    sets = (
        ("omat24_val90", "all", "OMat24 held-out", "ID"),
        ("matpes", "bulk", "MatPES bulk", "SHIFT"),
        ("matpes", "non_bulk", "MatPES non-bulk", "OOD"),
    )
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), sharex=True, sharey=True)
        for i, (ax, (folder, domain, title, distance)) in enumerate(zip(axes, sets)):
            rows = read_csv(results / "accuracy" / folder / "force_magnitude.csv")
            guide = np.logspace(-2, 2, 9)
            ax.plot(guide, 1e3 * guide, color=INK["muted"], lw=0.6, ls=(0, (1.2, 1.6)), zorder=1)
            for m in ("mace_omat0", "authors_envtok", "ours"):
                # not the bin from zero: most of its MatPES atoms have no force in DFT or the model
                sub = [
                    r
                    for r in rows
                    if r["model"] == m
                    and r["domain"] == domain
                    and int(r["n_atoms"]) >= 100
                    and float(r["ref_lo_eV_per_A"]) > 0
                ]
                lo = np.array([float(r["ref_lo_eV_per_A"]) for r in sub])
                hi = np.array([float(r["ref_hi_eV_per_A"]) for r in sub])
                x = np.sqrt(lo * hi)
                err = np.array([float(r["median_err_norm_meV_per_A"]) for r in sub])
                if m == "mace_omat0":
                    ax.plot(x, err, color=color(m), lw=0.9, ls=MODELS[m]["linestyle"], zorder=2)
                else:
                    ax.plot(x, err, color=color(m), lw=1.25, zorder=3)
                    mark(ax, x, err, m, ms=3.6, z=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(8e-3, 150)
            ax.set_ylim(12, 5000)
            ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
            ax.set_yticks([10, 100, 1000], ["10", "100", "1,000"])
            ax.minorticks_off()
            ax.set_xlabel("DFT force on the atom (eV Å$^{-1}$)")
            ax.set_title(title, loc="left", pad=4)
            chip(ax, distance)
            letter(ax, "abc"[i], x=-0.02 if i else -0.2)
        axes[0].text(0.2, 400, "error = force", fontsize=5.4, color=INK["muted"], ha="right", va="center")
        axes[0].set_ylabel("Median force error (meV Å$^{-1}$)")
        axes[0].legend(
            handles=[handle("ours"), handle("authors_envtok"), handle("mace_omat0")],
            loc="lower right",
            fontsize=5.6,
            handletextpad=0.4,
            borderaxespad=0.4,
        )
        fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.17, wspace=0.08)
        save(fig, out, "error_vs_force")


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
