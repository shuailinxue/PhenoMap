"""Plotting helpers shared by the Step 3 query tutorials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def set_plot_style() -> None:
    """Apply the established query-tutorial plotting defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


def load_rgb(path) -> np.ndarray:
    """Load an image as an RGB array."""
    return np.asarray(Image.open(path).convert("RGB"))


def _relative_score(score: np.ndarray) -> np.ndarray:
    finite = np.isfinite(score)
    low, high = float(score[finite].min()), float(score[finite].max())
    return np.clip((score - low) / (high - low + 1e-8), 0, 1)


def plot_query_score_map(
    score: np.ndarray,
    ax=None,
    title: str | None = None,
    background: np.ndarray | None = None,
    cmap: str = "magma",
    alpha: float = 0.55,
):
    """Plot a query score directly or blend its relative values over an image."""
    if ax is None:
        _, ax = plt.subplots()
    if background is None:
        artist = ax.imshow(score, cmap=cmap, vmin=0, vmax=1, origin="upper")
    else:
        relative = _relative_score(score)
        heat = (plt.get_cmap(cmap)(relative)[..., :3] * 255).astype(np.uint8)
        resized = np.asarray(Image.fromarray(background).resize((score.shape[1], score.shape[0]), Image.Resampling.LANCZOS))
        blended = (resized * (1 - alpha) + heat * alpha).astype(np.uint8)
        artist = ax.imshow(blended)
    if title is not None:
        ax.set_title(title, fontweight="bold")
    ax.axis("off")
    return artist


def plot_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    colors: np.ndarray,
    ax=None,
    title: str | None = None,
    alpha: float = 0.5,
):
    """Blend a categorical mask over an RGB image."""
    if ax is None:
        _, ax = plt.subplots()
    if mask.ndim == 3:
        mask = mask[..., 0]
    clipped = np.clip(mask.astype(int), 0, len(colors) - 1)
    overlay = colors[clipped]
    weights = mask.astype(bool).astype(float)[..., None] * alpha
    ax.imshow((image * (1 - weights) + overlay * weights).astype(np.uint8))
    if title is not None:
        ax.set_title(title, fontweight="bold")
    ax.axis("off")
    return ax


def plot_method_metric_bars(
    summary: pd.DataFrame,
    methods: Sequence[str],
    method_colors: Mapping[str, str],
    metric_specs: Sequence[tuple[str, str]],
    ax=None,
    title: str | None = None,
):
    """Plot grouped method means and standard errors for several metrics."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(metric_specs))
    width = 0.23
    offsets = np.linspace(-width, width, len(methods))
    for offset, method in zip(offsets, methods):
        means = [summary.loc[method, (metric, "mean")] for metric, _ in metric_specs]
        errors = [summary.loc[method, (metric, "sem")] for metric, _ in metric_specs]
        ax.bar(x + offset, means, width, yerr=errors, capsize=2, color=method_colors[method], edgecolor="white", linewidth=0.5, label=method)
    ax.set_xticks(x, [label for _, label in metric_specs])
    if title is not None:
        ax.set_title(title, fontweight="bold")
    return ax


def plot_binary_query_curves(
    localization: pd.DataFrame,
    curves: Mapping[str, Mapping[str, np.ndarray]],
    methods: Sequence[str],
    method_colors: Mapping[str, str],
    axes=None,
):
    """Plot paired cell-level ROC and precision–recall curves."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    metric_table = localization.set_index("method")
    for method in methods:
        curve = curves[method]
        axes[0].plot(curve["fpr"], curve["tpr"], color=method_colors[method], lw=1.8, label=f"{method}  AUC={metric_table.loc[method, 'roc_auc']:.3f}")
        axes[1].plot(curve["recall"], curve["precision"], color=method_colors[method], lw=1.8, label=f"{method}  AP={metric_table.loc[method, 'average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=0.8, color="#AAAAAA")
    axes[1].axhline(localization["positive_fraction"].iloc[0], ls="--", lw=0.8, color="#AAAAAA")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Cell-level ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Cell-level precision–recall")
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="lower right")
    return axes


def plot_concordance_intervals(
    concordance: pd.DataFrame,
    methods: Sequence[str],
    method_colors: Mapping[str, str],
    metric_specs: Sequence[tuple[str, str, tuple[float, float], float]],
    axes=None,
):
    """Plot concordance estimates and bootstrap confidence intervals."""
    if axes is None:
        _, axes = plt.subplots(1, len(metric_specs), figsize=(10.8, 3.2))
    axes = np.atleast_1d(axes)
    for panel, (metric, xlabel, xlim, reference) in zip(axes, metric_specs):
        table = concordance.loc[concordance["metric"].eq(metric)].set_index("method").loc[list(methods)]
        y = np.arange(len(methods))[::-1]
        panel.axvline(reference, color="#BBBBBB", lw=0.8, ls="--")
        for y_value, method in zip(y, methods):
            row = table.loc[method]
            panel.errorbar(row["estimate"], y_value, xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]], fmt="o", color=method_colors[method], ecolor=method_colors[method], markersize=6, capsize=2, linewidth=1.2)
        panel.set_xlim(*xlim); panel.set_ylim(-0.6, len(methods) - 0.4)
        panel.set_yticks(y); panel.set_yticklabels(methods); panel.set_xlabel(xlabel)
        panel.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    return axes
