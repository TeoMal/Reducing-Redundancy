#!/usr/bin/env python
"""Render the paper/README figures from the per-epoch CSV logs.

Reads whichever runs are present under ``outputs/`` and writes PNGs to
``docs/figures/``. Missing runs are skipped with a warning, so this is safe to
run while some experiments are still queued.

    python scripts/make_figures.py

Figures are styled for print: single-column width, no in-figure titles (the
caption carries them), light rules, and a zoom inset on the accuracy plot --
without it the 1-point differences that the whole study is about are a few
pixels tall.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = "docs/figures"

# Reference categorical palette, slots 1-4 in fixed order; line charts are the
# adjacent-pair case this ordering is validated for. Slots 3 and 4 sit below 3:1
# contrast on a light surface, so the README carries the results table alongside
# -- identity is never left to colour alone.
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
}
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd7"

# Print-figure defaults. Kept local to this script rather than a global style
# file so running it never perturbs anyone else's matplotlib settings.
PAPER_RC = {
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Plain text, not mathtext: "$c{=}1$" swallows the preceding space when
# matplotlib lays the label out, so the entries render inconsistently.
SELECTORS = [
    ("Full (baseline)", "outputs/imagenet_vit_b_16_full.csv", C["blue"]),
    ("TopK (k = 0.5)", "outputs/imagenet_vit_b_16_topk_two_pass.csv", C["orange"]),
    ("AdaptiveK (f = 2/3)", "outputs/imagenet_vit_b_16_adaptive_k_two_pass.csv", C["aqua"]),
    ("MeanAdaptive (c = 1)", "outputs/imagenet_vit_b_16_mean_adaptive_two_pass.csv", C["yellow"]),
]


def style(ax, xlabel, ylabel, grid_axis="both"):
    """Recessive axes: two spines, hairline grid, no in-figure title."""
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)


def load(runs):
    """Return [(label, dataframe, colour)] for the runs whose CSV exists."""
    out = []
    for label, path, colour in runs:
        if os.path.exists(path):
            out.append((label, pd.read_csv(path), colour))
        else:
            print(f"  skip (missing): {path}")
    return out


def save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", path)


def fig_accuracy(runs, fname):
    """Top-1 against epoch, with a zoom inset over the converged tail.

    Legend order follows final accuracy rather than the input order, so reading
    the legend top-to-bottom matches reading the curves at the right edge.
    """
    data = load(runs)
    if not data:
        print(f"  nothing to plot for {fname}")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)

    ranked = sorted(data, key=lambda t: -t[1].eval_acc_top1.iloc[-5:].mean())
    for label, d, colour in ranked:
        ax.plot(d.epoch, d.eval_acc_top1 * 100, color=colour, label=label, zorder=3)

    style(ax, "Epoch", "Top-1 accuracy (\\%)" if plt.rcParams["text.usetex"]
          else "Top-1 accuracy (%)")
    xmax = max(d.epoch.max() for _l, d, _c in data)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 80)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK,
              handlelength=1.6, borderaxespad=0.6)

    # Zoom on the converged tail: at full scale the gaps between methods are
    # about a pixel, which is precisely the quantity under study.
    lo_ep = int(xmax * 0.8)
    tail = [(d[d.epoch >= lo_ep].eval_acc_top1 * 100) for _l, d, _c in data]
    y_lo = min(s.min() for s in tail) - 0.4
    y_hi = max(s.max() for s in tail) + 0.4

    axin = ax.inset_axes([0.50, 0.12, 0.47, 0.40], facecolor=SURFACE)
    for label, d, colour in ranked:
        sel = d[d.epoch >= lo_ep]
        axin.plot(sel.epoch, sel.eval_acc_top1 * 100, color=colour, zorder=3)
    axin.set_xlim(lo_ep, xmax)
    axin.set_ylim(y_lo, y_hi)
    axin.grid(True, color=GRID, linewidth=0.4, zorder=0)
    axin.set_axisbelow(True)
    for s in axin.spines.values():
        s.set_color(INK2)
        s.set_linewidth(0.6)
    axin.tick_params(colors=INK2, labelsize=7, width=0.6, size=2)
    axin.set_title(f"final {xmax - lo_ep} epochs", fontsize=7.5, color=INK2, pad=3)
    ax.indicate_inset_zoom(axin, edgecolor=INK2, linewidth=0.6, alpha=0.55)

    save(fig, fname)


def fig_budget(runs, fname):
    """Examples kept and modelled compute, side by side on a shared x-axis."""
    data = load(runs)
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), facecolor=SURFACE,
                             sharex=True)
    for label, d, colour in data:
        axes[0].plot(d.epoch, d.effective_ratio * 100, color=colour, label=label)
        axes[1].plot(d.epoch, d.compute_ratio * 100, color=colour, label=label)
    style(axes[0], "Epoch", "Examples kept (%)")
    style(axes[1], "Epoch", "Compute vs.\\ baseline (%)"
          if plt.rcParams["text.usetex"] else "Compute vs. baseline (%)")
    for ax in axes:
        ax.set_ylim(0, 108)
        ax.set_xlim(0, max(d.epoch.max() for _l, d, _c in data))
    axes[0].legend(loc="lower left", frameon=False, labelcolor=INK,
                   handlelength=1.6, borderaxespad=0.6)
    fig.tight_layout(pad=0.4)
    save(fig, fname)


def fig_tradeoff(runs, fname):
    """Accuracy against the data budget.

    One hue plus text labels rather than four categorical colours: a scatter is
    an all-pairs form, where the categorical palette caps at three series.
    """
    data = load(runs)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.8), facecolor=SURFACE)
    pts = sorted(((d.effective_ratio.mean() * 100, d.eval_acc_top1.max() * 100, label)
                  for label, d, _ in data), key=lambda t: t[1])
    for x, y, _label in pts:
        ax.scatter([x], [y], s=42, color=C["blue"], zorder=3,
                   edgecolor=SURFACE, linewidth=1.0)

    style(ax, "Mean examples used (%)", "Best top-1 accuracy (%)")
    ax.set_xlim(12, 118)
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - span * 0.06, hi + span * 0.06)
    lo, hi = ax.get_ylim()

    # The interesting selectors land within a fraction of a point of each other,
    # so two-line labels anchored at the marker overprint. Push them to a minimum
    # vertical separation and draw a leader back to the marker when moved.
    min_gap = (hi - lo) * 0.16
    placed: list[float] = []
    for _x, y, _label in pts:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
    # Grow the axes to fit the label stack rather than sliding it down -- sliding
    # pushes the lowest label off the bottom, where it is silently invisible.
    pad = min_gap * 0.6
    ax.set_ylim(min(lo, min(placed) - pad), max(hi, max(placed) + pad))
    lo, hi = ax.get_ylim()

    for (x, y_true, label), y_lab in zip(pts, placed):
        # Default to the right of the marker, but flip left when that would run
        # the text over a neighbouring point (TopK and AdaptiveK sit 3.5 points
        # of data and 0.17pp of accuracy apart), or when the marker is already
        # near the right edge.
        blocked = any(
            0 < ox - x < 14 and abs(oy - y_lab) < min_gap * 0.6
            for ox, oy, _ol in pts if ox != x
        )
        right = x > 75 or blocked
        dx = -9 if right else 9
        ax.annotate(
            f"{label}\n{y_true:.2f}% at {x:.0f}% of data",
            (x, y_lab), color=INK, fontsize=8, va="center",
            ha="right" if right else "left",
            xytext=(dx, 0), textcoords="offset points",
        )
        if abs(y_lab - y_true) > (hi - lo) * 0.01:
            ax.plot([x, x + (dx / 9) * 1.5], [y_true, y_lab], color=INK2,
                    linewidth=0.5, alpha=0.6, zorder=2)

    fig.tight_layout(pad=0.4)
    save(fig, fname)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        print("selector accuracy:")
        fig_accuracy(SELECTORS, "selector_accuracy.png")
        print("budget:")
        fig_budget(SELECTORS, "selector_budget.png")
        print("trade-off:")
        fig_tradeoff(SELECTORS, "accuracy_vs_data.png")


if __name__ == "__main__":
    main()
