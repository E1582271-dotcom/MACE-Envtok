"""Layout checks run on every figure before it is saved.

Two rules from the figure style: nothing may touch (text against text, against
data lines and markers, against reference lines or the frame; a legend against
data), and sibling panels must line up (same row: same top and bottom; same
column: same left edge). Gridlines may run behind text; a reference line may meet
the frame. The checks work on the drawn figure in display coordinates.
"""

from __future__ import annotations


def _text_boxes(fig, renderer):
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    ticks = {}
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            span = hi - lo
            for tick in axis.get_major_ticks():
                drawn = lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span
                for lab in (tick.label1, tick.label2):
                    ticks[id(lab)] = drawn and lab.get_visible()
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or not ticks.get(id(t), True):
            continue
        bb = t.get_window_extent(renderer)
        if t.get_bbox_patch() is not None:
            bb = Bbox.union([bb, t.get_bbox_patch().get_window_extent(renderer)])
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb))
    return out


def collisions(fig, pad_px: float = 0.5) -> list[str]:
    """Every pair of elements that touch, as readable strings (empty list: clean)."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = _text_boxes(fig, r)
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    legend_parts = {id(a) for lg in legends for a in lg.findobj()}
    found = []

    def name(t):
        return repr(t.get_text().replace("\n", " ")[:40])

    # text against text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1].padded(-pad_px), texts[j][1].padded(-pad_px)
            if a.overlaps(b):
                found.append(f"text {name(texts[i][0])} touches text {name(texts[j][0])}")

    for ax in fig.axes:
        axbox = ax.get_window_extent(r)
        lines = [ln for ln in ax.get_lines() if ln.get_visible() and id(ln) not in legend_parts]
        regions = [(f"text {name(t)}", bb) for t, bb in texts if id(t) not in legend_parts]
        regions += [
            (f"legend of {ax.get_title(loc='left')[:30]!r}", lg.get_window_extent(r))
            for lg in legends
            if lg.axes is ax
        ]
        for what, bb in regions:
            inner = Bbox.intersection(bb.padded(-pad_px), axbox)
            if inner is None or inner.width <= 0 or inner.height <= 0:
                continue
            for ln in lines:
                path = ln.get_transform().transform_path(ln.get_path()).cleaned(remove_nans=True)
                verts = path.vertices
                if len(verts) == 0:
                    continue
                if ln.get_linestyle() not in ("None", "none", "", " ") and path.intersects_bbox(
                    inner, filled=False
                ):
                    found.append(f"{what} touches a line in {ax.get_title(loc='left')[:30]!r}")
                    continue
                if ln.get_marker() not in (None, "None", "none", ""):
                    rad = ln.get_markersize() * fig.dpi / 72 / 2
                    grown = inner.padded(rad)
                    inside = (
                        (verts[:, 0] >= grown.x0)
                        & (verts[:, 0] <= grown.x1)
                        & (verts[:, 1] >= grown.y0)
                        & (verts[:, 1] <= grown.y1)
                    )
                    inside &= (
                        (verts[:, 0] >= axbox.x0)
                        & (verts[:, 0] <= axbox.x1)
                        & (verts[:, 1] >= axbox.y0)
                        & (verts[:, 1] <= axbox.y1)
                    )
                    if inside.any():
                        found.append(f"{what} touches a marker in {ax.get_title(loc='left')[:30]!r}")
            # text inside the panel must not cross the frame
            if what.startswith("text"):
                edges = (axbox.x0, axbox.x1)
                hedges = (axbox.y0, axbox.y1)
                full = bb.padded(-pad_px)
                if (
                    any(full.x0 < x < full.x1 for x in edges) and full.y1 > axbox.y0 and full.y0 < axbox.y1
                ) or (
                    any(full.y0 < y < full.y1 for y in hedges) and full.x1 > axbox.x0 and full.x0 < axbox.x1
                ):
                    found.append(f"{what} crosses the frame of {ax.get_title(loc='left')[:30]!r}")
    return sorted(set(found))


def misaligned(fig, tol_px: float = 0.5) -> list[str]:
    """Panels of one subplot grid that do not share their row's top/bottom or column's left edge."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    grids = {}
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None:
            continue
        grids.setdefault(id(spec.get_gridspec()), []).append((ax, spec, ax.get_window_extent(r)))
    out = []
    for members in grids.values():
        for i, (_, sa, ba) in enumerate(members):
            for _, sb, bb in members[i + 1 :]:
                if sa.rowspan == sb.rowspan and (abs(ba.y0 - bb.y0) > tol_px or abs(ba.y1 - bb.y1) > tol_px):
                    out.append(f"row {sa.rowspan.start}: panels differ in top/bottom")
                if sa.colspan == sb.colspan and abs(ba.x0 - bb.x0) > tol_px:
                    out.append(f"column {sa.colspan.start}: panels differ in left edge")
    return sorted(set(out))


def check(fig) -> list[str]:
    return collisions(fig) + misaligned(fig)
