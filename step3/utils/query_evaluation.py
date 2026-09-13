"""Evaluation utilities for morphology-based phenotype queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import center_of_mass
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def orient_molecular_scores_from_source(
    source_expression_path: str | Path,
    source_score_path: str | Path,
    target_expression_path: str | Path,
    target_raw_scores: pd.Series,
    *,
    n_genes_per_tail: int = 15,
) -> tuple[pd.Series, dict[str, float | int | bool]]:
    """Orient target molecular scores using a frozen source-derived gene anchor."""
    source_expression = pd.read_csv(source_expression_path, index_col=0)
    source_expression.index = source_expression.index.astype(str)
    source_scores = pd.read_parquet(source_score_path).iloc[:, 0]
    source_scores.index = source_scores.index.astype(str)
    common_source = source_expression.index.intersection(source_scores.index)
    correlations = source_expression.loc[common_source].rank(pct=True).corrwith(
        source_scores.loc[common_source].rank(pct=True)
    )
    weights = pd.concat(
        [
            correlations.nsmallest(n_genes_per_tail),
            correlations.nlargest(n_genes_per_tail),
        ]
    ).sort_values()

    target_expression = pd.read_csv(target_expression_path, index_col=0)
    target_expression.index = target_expression.index.astype(str)
    genes = weights.index.intersection(target_expression.columns)
    standardized = target_expression.loc[:, genes]
    standardized = (standardized - standardized.mean()) / standardized.std().replace(0, 1)
    anchor = standardized.mul(weights.loc[genes], axis=1).sum(axis=1)
    anchor /= weights.loc[genes].abs().sum()

    raw = target_raw_scores.copy()
    raw.index = raw.index.astype(str)
    common_target = raw.index.intersection(anchor.index)
    correlation = float(
        spearmanr(raw.loc[common_target], anchor.loc[common_target]).statistic
    )
    if not np.isfinite(correlation) or correlation == 0:
        raise ValueError("The source molecular anchor could not orient target scores")
    multiplier = 1 if correlation > 0 else -1
    oriented = (raw * multiplier).rename("molecular_pr")
    diagnostics = {
        "multiplier": multiplier,
        "raw_anchor_spearman": correlation,
        "n_matched_cells": int(len(common_target)),
        "uses_target_cell_type_labels": False,
        "uses_morphology_query_scores": False,
    }
    return oriented, diagnostics


def load_cell_coordinates(
    annotation_path: str | Path,
    *,
    coordinate_cache: str | Path | None = None,
    mask_path: str | Path | None = None,
    cell_type_col: str = "cell_type",
) -> tuple[pd.DataFrame, tuple[int, int]]:
    """Load reference labels and cell centers, preferring an existing cache."""
    annotation = pd.read_csv(annotation_path, index_col=0)
    annotation.index = annotation.index.astype(str)

    if coordinate_cache is not None and Path(coordinate_cache).exists():
        cached = pd.read_csv(coordinate_cache, index_col=0)[[cell_type_col, "x", "y"]]
        cached.index = cached.index.astype(str)
        cells = annotation[[cell_type_col]].join(cached[["x", "y"]], how="inner")
        if mask_path is None:
            raise ValueError("mask_path is required to recover the coordinate-frame shape")
        with tifffile.TiffFile(mask_path) as tif:
            shape = tuple(int(value) for value in tif.series[0].shape[:2])
        return cells.dropna(subset=["x", "y"]), shape

    if mask_path is None:
        raise ValueError("mask_path is required when no coordinate cache is available")
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[..., 0]
    cell_ids = annotation.index.to_numpy(dtype=int)
    centers_yx = np.asarray(center_of_mass(mask, labels=mask, index=cell_ids), dtype=float)
    centers = pd.DataFrame(
        {"x": centers_yx[:, 1], "y": centers_yx[:, 0]}, index=annotation.index
    )
    return annotation[[cell_type_col]].join(centers).dropna(subset=["x", "y"]), mask.shape


def sample_score_maps(
    cells: pd.DataFrame,
    score_maps: Mapping[str, str | Path],
    coordinate_shape: tuple[int, int],
) -> pd.DataFrame:
    """Sample downscaled score maps at cell centers in the source coordinate frame."""
    result = cells.copy()
    source_height, source_width = coordinate_shape
    for method, map_path in score_maps.items():
        score_map = np.load(map_path, mmap_mode="r")
        y_index = np.clip(
            np.rint(result["y"].to_numpy() * score_map.shape[0] / source_height).astype(int),
            0,
            score_map.shape[0] - 1,
        )
        x_index = np.clip(
            np.rint(result["x"].to_numpy() * score_map.shape[1] / source_width).astype(int),
            0,
            score_map.shape[1] - 1,
        )
        result[method] = np.asarray(score_map[y_index, x_index], dtype=np.float32)
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=list(score_maps))


def top_fraction_composition(
    cells: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    fraction: float = 0.05,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    """Calculate reference-label composition among each method's highest scores."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    rows = []
    for method in score_columns:
        threshold = float(cells[method].quantile(1 - fraction))
        selected = cells.loc[cells[method] >= threshold, cell_type_col]
        counts = selected.value_counts()
        for cell_type, count in counts.items():
            rows.append(
                {
                    "method": method,
                    "cell_type": cell_type,
                    "n_top_cells": int(count),
                    "fraction": float(count / len(selected)),
                    "threshold": threshold,
                    "top_fraction": fraction,
                }
            )
    return pd.DataFrame(rows)


def binary_query_performance(
    cells: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    positive_cell_type: str = "Cancer_epithelial",
    cell_type_col: str = "cell_type",
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Compute continuous-score ROC and precision-recall performance."""
    labels = cells[cell_type_col].astype(str).eq(positive_cell_type).to_numpy()
    if labels.all() or (~labels).all():
        raise ValueError("Both positive and negative cells are required")
    rows = []
    curves = {}
    for method in score_columns:
        scores = cells[method].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        rows.append(
            {
                "method": method,
                "n_cells": int(len(labels)),
                "n_positive": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
            }
        )
        curves[method] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
        }
    return pd.DataFrame(rows), curves


def aggregate_spatial_blocks(
    cells: pd.DataFrame,
    columns: Sequence[str],
    *,
    block_size: int = 1024,
) -> pd.DataFrame:
    """Summarize cell-level values by median within square spatial blocks."""
    frame = cells.loc[:, ["x", "y", *columns]].copy()
    frame["block_x"] = np.floor(frame["x"] / block_size).astype(int)
    frame["block_y"] = np.floor(frame["y"] / block_size).astype(int)
    return frame.groupby(["block_x", "block_y"], sort=True)[list(columns)].median()


def _block_sufficient_statistics(
    x: np.ndarray,
    y: np.ndarray,
    block_x: pd.Series,
    block_y: pd.Series,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "block_x": block_x.to_numpy(),
            "block_y": block_y.to_numpy(),
        }
    )
    frame["xx"] = frame["x"] ** 2
    frame["yy"] = frame["y"] ** 2
    frame["xy"] = frame["x"] * frame["y"]
    return frame.groupby(["block_x", "block_y"], sort=True).agg(
        n=("x", "size"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxx=("xx", "sum"),
        syy=("yy", "sum"),
        sxy=("xy", "sum"),
    ).to_numpy(dtype=float)


def _pearson_from_sums(values: np.ndarray) -> float:
    n, sx, sy, sxx, syy, sxy = values
    numerator = sxy - sx * sy / n
    denominator = np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
    return float(numerator / denominator)


def spatial_block_bootstrap(
    cells: pd.DataFrame,
    blocks: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    molecular_col: str = "molecular_pr",
    block_size: int = 1024,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate S3 concordance metrics and block-bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    sampled_blocks = rng.integers(0, n_blocks, size=(n_bootstrap, n_blocks))
    molecular_blocks = blocks[molecular_col].to_numpy(dtype=float)
    low, high = np.quantile(molecular_blocks, [0.2, 0.8])
    extremes = (molecular_blocks <= low) | (molecular_blocks >= high)
    extreme_labels = molecular_blocks[extremes] >= high

    block_x = np.floor(cells["x"] / block_size).astype(int)
    block_y = np.floor(cells["y"] / block_size).astype(int)
    cell_molecular = cells[molecular_col].to_numpy(dtype=float)
    rows = []

    for method in score_columns:
        score_blocks = blocks[method].to_numpy(dtype=float)
        cell_scores = cells[method].to_numpy(dtype=float)
        sufficient = _block_sufficient_statistics(
            cell_scores, cell_molecular, block_x, block_y
        )

        estimates = {
            "block_spearman": float(spearmanr(score_blocks, molecular_blocks).statistic),
            "cell_pearson": float(pearsonr(cell_scores, cell_molecular).statistic),
            "extreme_quintile_auc": float(
                roc_auc_score(extreme_labels, score_blocks[extremes])
            ),
        }
        draws = {metric: np.empty(n_bootstrap) for metric in estimates}
        for draw, sampled in enumerate(sampled_blocks):
            sampled_scores = score_blocks[sampled]
            sampled_molecular = molecular_blocks[sampled]
            draws["block_spearman"][draw] = spearmanr(
                sampled_scores, sampled_molecular
            ).statistic
            draws["cell_pearson"][draw] = _pearson_from_sums(
                sufficient[sampled].sum(axis=0)
            )
            draw_low, draw_high = np.quantile(sampled_molecular, [0.2, 0.8])
            draw_extremes = (sampled_molecular <= draw_low) | (
                sampled_molecular >= draw_high
            )
            draw_labels = sampled_molecular[draw_extremes] >= draw_high
            draws["extreme_quintile_auc"][draw] = (
                roc_auc_score(draw_labels, sampled_scores[draw_extremes])
                if np.unique(draw_labels).size == 2
                else np.nan
            )

        for metric, estimate in estimates.items():
            ci_low, ci_high = np.nanquantile(draws[metric], [0.025, 0.975])
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_cells": int(len(cells)),
                    "n_spatial_blocks": int(n_blocks),
                    "block_size": int(block_size),
                    "n_bootstrap": int(n_bootstrap),
                    "seed": int(seed),
                    "resampling_scheme": "paired_spatial_blocks_v1",
                }
            )
    return pd.DataFrame(rows)
