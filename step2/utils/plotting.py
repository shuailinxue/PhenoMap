"""Plotting helpers shared by the Step 2 tutorials."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Apply the established tutorial plotting defaults."""
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    sns.set_theme(style="white", context="notebook")


def plot_spatial_phenotype_scores(
    cells: pd.DataFrame,
    score_col: str = "beta",
    background=None,
    ax=None,
    title: str | None = "Spatial distribution of phenotype-associated cells",
    clip_abs: float | None = 2.0,
    point_size: float = 1.0,
    cmap: str = "RdBu_r",
    center: float | None = 0.0,
    hide_axes: bool = True,
    alpha: float = 0.9,
):
    """Plot prepared cell-level phenotype scores, optionally over histology."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8), dpi=180)
    if background is not None:
        ax.imshow(background)
    scores = cells[score_col].astype(float)
    if clip_abs is not None:
        scores = scores.clip(-float(clip_abs), float(clip_abs))
    norm = None
    if center is not None:
        limit = float(clip_abs) if clip_abs is not None else float(max(abs(scores.min()), abs(scores.max())))
        norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=center, vmax=limit)
    points = ax.scatter(
        cells["x"], cells["y"], c=scores,
        norm=norm, cmap=cmap, s=point_size, linewidths=0, alpha=alpha, rasterized=True,
    )
    if title is not None:
        ax.set_title(title)
    if hide_axes:
        ax.axis("off")
    return points


def plot_celltype_risk_sankey(
    flow: pd.DataFrame,
    left_nodes: Sequence[str],
    risk_nodes: Sequence[str],
    celltype_colors: Mapping[str, str],
    risk_colors: Mapping[str, str],
    ax=None,
    title: str | None = None,
):
    """Draw a static cell-type-to-risk Sankey from a prepared flow table."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8.5))
    total = flow["value"].sum()
    gap = 0.02
    available_height = 1 - gap * (max(len(left_nodes), len(risk_nodes)) - 1)
    scale = available_height / total

    def node_spans(nodes, column):
        totals = flow.groupby(column, observed=True)["value"].sum()
        spans, top = {}, 1.0
        for node in nodes:
            height = totals[node] * scale
            spans[node] = [top - height, top]
            top -= height + gap
        return spans

    left_spans = node_spans(left_nodes, "CellType")
    right_spans = node_spans(risk_nodes, "RiskStrata")
    left_cursor = {node: span[1] for node, span in left_spans.items()}
    right_cursor = {node: span[1] for node, span in right_spans.items()}
    x_left, x_right, node_width = 0.08, 0.89, 0.025

    for left in left_nodes:
        for right in risk_nodes:
            selected = (flow["CellType"] == left) & (flow["RiskStrata"].astype(str) == right)
            value = int(flow.loc[selected, "value"].sum())
            if value == 0:
                continue
            height = value * scale
            y0_top, y0_bottom = left_cursor[left], left_cursor[left] - height
            y1_top, y1_bottom = right_cursor[right], right_cursor[right] - height
            left_cursor[left], right_cursor[right] = y0_bottom, y1_bottom
            vertices = [
                (x_left + node_width, y0_top),
                (0.38, y0_top), (0.62, y1_top), (x_right, y1_top),
                (x_right, y1_bottom),
                (0.62, y1_bottom), (0.38, y0_bottom), (x_left + node_width, y0_bottom),
                (x_left + node_width, y0_top),
            ]
            codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO] + [MplPath.CURVE4] * 3 + [MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=celltype_colors[left], edgecolor="none", alpha=0.60))

    for node, (bottom, top) in left_spans.items():
        ax.add_patch(Rectangle((x_left, bottom), node_width, top - bottom, color=celltype_colors[node]))
        ax.text(x_left - 0.012, (bottom + top) / 2, node, ha="right", va="center")
    for node, (bottom, top) in right_spans.items():
        ax.add_patch(Rectangle((x_right, bottom), node_width, top - bottom, color=risk_colors[node]))
        ax.text(x_right + node_width + 0.012, (bottom + top) / 2, node, ha="left", va="center")
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    return ax


def plot_kaplan_meier_panel(
    analysis: pd.DataFrame,
    event_col: str,
    ax=None,
    title: str | None = None,
    logrank_p: float | None = None,
    max_months: float = 150,
):
    """Draw the established high-/low-risk Kaplan–Meier panel."""
    from lifelines import KaplanMeierFitter

    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.2))
    groups = [("High", "High risk", "#C84A47"), ("Low", "Low risk", "#4C8DAE")]
    for group, label, color in groups:
        subset = analysis.loc[analysis["group"].eq(group)]
        KaplanMeierFitter(label=f"{label} (n={len(subset)})").fit(
            subset["time_months"], subset[event_col]
        ).plot_survival_function(ax=ax, ci_show=False, color=color, lw=1.7)
    if logrank_p is not None:
        ax.text(0.97, 0.08, f"Log-rank P = {logrank_p:.2g}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlim(0, max_months)
    ax.set_ylim(0.5, 1.03)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    if title is not None:
        ax.set_title(title)
    return ax
