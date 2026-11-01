"""Figures: the layout checker finds what it should, and every figure passes it."""

from pathlib import Path

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from envtok_mace.plots import audit, figures  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_checker_finds_text_on_a_line_and_on_another_text():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.text(0.5, 0.5, "on the line", ha="center", va="center")
    ax.text(0.05, 0.9, "first label")
    ax.text(0.06, 0.9, "second label")
    found = audit.collisions(fig)
    plt.close(fig)
    assert any("on the line" in f and "line" in f.split("touches")[1] for f in found)
    assert any("first label" in f and "second label" in f for f in found)


def test_checker_passes_a_clean_figure():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.text(0.95, 0.05, "clear of the line", ha="right", va="bottom")
    assert audit.collisions(fig) == []
    plt.close(fig)


def test_checker_finds_misaligned_rows():
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    pos = axes[1].get_position()
    axes[1].set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height - 0.05])
    assert audit.misaligned(fig)
    plt.close(fig)


def test_every_figure_renders_without_layout_problems(tmp_path):
    figures.PROBLEMS.clear()
    for name, draw in figures.FIGURES.items():
        draw(RESULTS, tmp_path)
        for ext in ("svg", "pdf", "png"):
            assert (tmp_path / f"{name}.{ext}").stat().st_size > 0
    assert {k: v for k, v in figures.PROBLEMS.items() if v} == {}
    assert set(figures.PROBLEMS) == set(figures.FIGURES)  # each figure saved under its own name
    # every font declaration in the SVGs falls back to a sans-serif face
    import re

    for svg in tmp_path.glob("*.svg"):
        fonts = re.findall(r"font-family: ?([^;\"]*)", svg.read_text())
        assert fonts and all(f.rstrip().endswith("sans-serif") for f in fonts), svg.name
