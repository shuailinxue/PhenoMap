"""Marker selection for the seven broad breast-tissue cell classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


DEFAULT_MARKER_GENES: dict[str, list[str]] = {
    "B_cells": ["MZB1", "MS4A1", "CD83", "BANK1", "ITM2C", "SEC11C"],
    "Cancer_epithelial": ["ELF3", "S100A14", "CLDN4", "KRT7", "KRT8", "AGR3", "ANKRD30A", "EPCAM", "CCND1", "CD9"],
    "Endothelial": ["RAMP2", "VWF", "AQP1", "PECAM1", "CLEC14A", "CD93", "TCF4", "PDK4", "STC1", "MMRN2"],
    "Myeloid": ["LYZ", "C1QA", "C1QC", "TYROBP", "APOC1", "FCER1G", "AIF1", "CD68", "CD14"],
    "Normal_epithelial": ["KRT14", "KRT15", "PTN", "TACSTD2", "KRT5", "SFRP1", "DST", "ACTG2"],
    "Stromal": ["LUM", "ACTA2", "POSTN", "CCDC80", "SFRP4", "PTGDS", "MMP2", "FBLN1", "CXCL12", "MYH11"],
    "T_cells": ["CCL5", "IL7R", "CD3E", "CXCR4", "CD69", "CD3D", "GZMA", "PTPRC", "DUSP2", "TRAC"],
}


def map_reference_cell_type(label: object) -> str | None:
    """Map a detailed breast-atlas label to a broad public cell class."""
    value = str(label).lower()
    rules = (
        (("cancer", "malignant", "carcinoma", "tumor", "neoplastic"), "Cancer_epithelial"),
        (("normal", "luminal", "basal", "myoepithelial", "epithelial"), "Normal_epithelial"),
        (("fibroblast", "caf", "stroma", "pvl", "pericyte"), "Stromal"),
        (("endo",), "Endothelial"),
        (("t cell", "t-cell", "t-cells", "cd4", "cd8", "nkt"), "T_cells"),
        (("b cell", "b-cell", "b-cells", "plasma", "plasmablast"), "B_cells"),
        (("myeloid", "macrophage", "monocyte", "dendritic", "dc"), "Myeloid"),
    )
    return next((target for terms, target in rules if any(term in value for term in terms)), None)


def select_marker_genes(
    marker_table: str | Path | pd.DataFrame,
    measured_expression: str | Path | pd.DataFrame,
    *,
    detection_fraction: float = 0.05,
    adjusted_p_max: float = 0.05,
    specificity_margin: float = 0.1,
    minimum_log_fold_change: float = 1.0,
    maximum_markers: int = 10,
    candidate_markers: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Select detected, significant and cell-type-specific marker genes."""
    if isinstance(marker_table, (str, Path)):
        markers = pd.read_excel(marker_table, skiprows=3, usecols=range(7), sheet_name="Supplementary Table 9")
    else:
        markers = marker_table.copy()
    if isinstance(measured_expression, (str, Path)):
        expression = pd.read_csv(measured_expression, index_col=0)
    else:
        expression = measured_expression.copy()
    numeric = expression.select_dtypes(include="number")
    detected = set(numeric.columns[(numeric != 0).mean() >= detection_fraction].astype(str).str.upper())

    markers["cell_type"] = markers["cluster"].map(map_reference_cell_type)
    markers["gene"] = markers["gene"].astype(str).str.upper()
    markers["avg_logFC"] = pd.to_numeric(markers["avg_logFC"], errors="coerce")
    markers["p_val_adj"] = pd.to_numeric(markers["p_val_adj"], errors="coerce")
    candidates = markers.loc[
        markers["cell_type"].notna()
        & markers["gene"].isin(detected)
        & markers["p_val_adj"].lt(adjusted_p_max)
    ].copy()

    keep = []
    for _, group in candidates.groupby("gene"):
        ordered = group.sort_values("avg_logFC", ascending=False)
        if len(ordered) == 1 or ordered.iloc[0]["avg_logFC"] - ordered.iloc[1]["avg_logFC"] > specificity_margin:
            keep.append(ordered.index[0])
    exclusive = candidates.loc[keep]
    selected: dict[str, list[str]] = {}
    for cell_type, group in exclusive.groupby("cell_type"):
        genes = group.loc[group["avg_logFC"].gt(minimum_log_fold_change)].sort_values("avg_logFC", ascending=False)["gene"]
        selected[str(cell_type)] = genes.head(maximum_markers).tolist()
    if candidate_markers is not None:
        selected = {
            cell_type: [gene for gene in genes if gene in set(selected.get(cell_type, []))]
            for cell_type, genes in candidate_markers.items()
        }
    return selected


def filter_available_markers(marker_genes: Mapping[str, Sequence[str]], genes: Sequence[str]) -> dict[str, list[str]]:
    available = set(map(str, genes))
    return {cell_type: [gene for gene in markers if gene in available] for cell_type, markers in marker_genes.items() if any(gene in available for gene in markers)}
