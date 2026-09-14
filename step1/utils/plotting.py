"""Plotting helpers shared by the Step 1 tutorials."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Apply the established tutorial plotting defaults."""
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    sns.set_theme(style="white", context="notebook")


def _finish(fig, output_path: str | Path | None, show: bool) -> Path | None:
    path = None if output_path is None else Path(output_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    return path


def _scale(values: Sequence[float], vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.clip((values - vmin) / (vmax - vmin if vmax > vmin else 1.0), 0, 1)


def plot_marker_gene_reconstruction(
    spot_df: pd.DataFrame,
    cell_coords: pd.DataFrame,
    true_aligned: pd.DataFrame,
    pred_aligned: pd.DataFrame,
    marker_genes: Sequence[str],
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot spot, ground-truth and predicted expression for marker genes."""
    marker_genes = list(marker_genes)
    if not marker_genes:
        raise ValueError("No marker genes were provided for plotting.")

    fig, axes = plt.subplots(len(marker_genes), 3, figsize=(11.5, 3.7 * len(marker_genes)), squeeze=False)
    headings = ["Spot-level expression", "Ground truth", "PhenoMap prediction"]
    for row, gene in enumerate(marker_genes):
        values = [
            spot_df[gene].to_numpy(dtype=float),
            true_aligned[gene].to_numpy(dtype=float),
            pred_aligned[gene].to_numpy(dtype=float),
        ]
        finite = np.concatenate([v[np.isfinite(v)] for v in values])
        vmin, vmax = np.percentile(finite, [1, 99])
        coordinates = [(spot_df["x"], spot_df["y"]), (cell_coords["x"], cell_coords["y"]), (cell_coords["x"], cell_coords["y"])]
        sizes = [4, 0.18, 0.18]
        for column, (ax, vals, coords, size) in enumerate(zip(axes[row], values, coordinates, sizes)):
            ax.scatter(*coords, c=_scale(vals, vmin, vmax), cmap="viridis", s=size, edgecolor="none", rasterized=column > 0)
            ax.set_title(headings[column] if row == 0 else "")
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.axis("off")
        axes[row, 0].set_ylabel(gene, fontweight="bold")

    fig.suptitle("Marker gene spatial reconstruction in HBC1", y=0.995, fontweight="bold")
    fig.tight_layout()
    path = _finish(fig, output_path, show)
    return path if path is not None else fig


def plot_spatial_cell_type_maps(
    metadata: pd.DataFrame,
    label_series_by_panel: Mapping[str, pd.Series],
    cell_types: Sequence[str],
    cell_colors: Mapping[str, str],
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot side-by-side spatial cell-type maps with a shared legend."""
    meta = metadata.copy()
    meta["display_y"] = meta["center_y"].max() - meta["center_y"]
    fig, axes = plt.subplots(1, len(label_series_by_panel), figsize=(10.8, 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, labels) in zip(axes, label_series_by_panel.items()):
        labels = labels.loc[meta.index]
        for cell_type in cell_types:
            index = labels.index[labels.eq(cell_type)]
            if len(index):
                coords = meta.loc[index]
                ax.scatter(coords["center_x"], coords["display_y"], s=0.35, c=cell_colors[cell_type], label=cell_type, alpha=0.85, linewidths=0, rasterized=True)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, markerscale=8)
    fig.tight_layout()
    path = _finish(fig, output_path, show)
    return path if path is not None else fig


def _ordered_heatmap(expr, annotation, categories):
    common = expr.index.intersection(annotation.index)
    expr, annotation = expr.loc[common], annotation.loc[common]
    ordered, boundaries, total = [], [], 0
    for category in categories:
        cells = annotation.index[annotation["cell_type"].eq(category)].tolist()
        if cells:
            ordered.extend(cells)
            total += len(cells)
            boundaries.append(total)
    expr, annotation = expr.loc[ordered], annotation.loc[ordered]
    matrix = expr.to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0.1] = 0.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = np.log1p(matrix / row_sums * 1e4)
    mins, maxs = matrix.min(axis=0), matrix.max(axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    return ((matrix - mins) / denom).T, annotation, boundaries


def plot_expression_annotation_heatmaps(
    true_expr: pd.DataFrame,
    pred_expr: pd.DataFrame,
    true_annotation: pd.DataFrame,
    pred_annotation: pd.DataFrame,
    marker_dict: Mapping[str, Sequence[str]],
    cell_colors: Mapping[str, str],
    output_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Plot paired expression heatmaps ordered by annotated cell type."""
    categories = list(marker_dict) + ["Unknown"]
    markers = list(dict.fromkeys(gene for genes in marker_dict.values() for gene in genes))
    markers = [gene for gene in markers if gene in true_expr and gene in pred_expr]
    true_matrix, true_ann, true_bounds = _ordered_heatmap(true_expr[markers], true_annotation, categories)
    pred_matrix, pred_ann, pred_bounds = _ordered_heatmap(pred_expr[markers], pred_annotation, categories)
    legend_types = [category for category in categories if category in cell_colors]
    codes = {category: i for i, category in enumerate(legend_types)}
    cmap = ListedColormap([cell_colors[category] for category in legend_types])

    fig = plt.figure(figsize=(11.0, 4.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035], height_ratios=[1, 0.035], wspace=0.08, hspace=0.07)
    main_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    bar_axes = [fig.add_subplot(gs[1, 0], sharex=main_axes[0]), fig.add_subplot(gs[1, 1], sharex=main_axes[1])]
    image = None
    for ax, bar_ax, matrix, annotation, boundaries, title in zip(main_axes, bar_axes, [true_matrix, pred_matrix], [true_ann, pred_ann], [true_bounds, pred_bounds], ["Ground truth", "PhenoMap"]):
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1, rasterized=True)
        fallback = codes.get("Unknown", 0)
        category_codes = annotation["cell_type"].map(codes).fillna(fallback).to_numpy()[None, :]
        bar_ax.imshow(category_codes, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(codes) - 1)
        for boundary in boundaries[:-1]:
            ax.axvline(boundary - 0.5, color="black", linewidth=0.8)
            bar_ax.axvline(boundary - 0.5, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([]); bar_ax.set_xticks([]); bar_ax.set_yticks([])
        for spine in bar_ax.spines.values():
            spine.set_visible(False)
    main_axes[0].set_ylabel("Genes", fontsize=13)
    fig.colorbar(image, cax=fig.add_subplot(gs[0, 2]), label="Scaled expression")
    handles = [Patch(facecolor=cell_colors[c], edgecolor="none", label=c.replace("_", " ")) for c in legend_types]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=7, frameon=False, fontsize=9)
    path = _finish(fig, output_path, show)
    return {"path": path, "marker_genes": markers, "ground_truth_cells": true_matrix.shape[1], "phenomap_cells": pred_matrix.shape[1]}


def plot_major_class_metric_bars(
    metric_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: str | Path | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = True,
):
    """Plot one metric per major cell class."""
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
    else:
        fig = ax.figure
    sns.barplot(data=metric_df, x="major_class", y=metric_col, color="#DE5253", ax=ax)
    ax.axhline(0, color="#777777", linewidth=0.8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set(xlabel="", ylabel=ylabel, title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path = _finish(fig, output_path, show) if owns_figure else None
    return path if path is not None else ax
