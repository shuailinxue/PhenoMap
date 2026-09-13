
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def orient_scores(
    scores: pd.Series,
    annotation_csv: str | Path,
    cell_type_col: str,
    positive_cell_types: list[str],
    statistic: str = "median",
    min_cells: int = 20,
) -> tuple[pd.Series, dict[str, Any]]:
    if statistic not in {"mean", "median"}:
        raise ValueError("orientation statistic must be 'mean' or 'median'")
    if not positive_cell_types:
        raise ValueError("positive_cell_types must not be empty")

    annotation = pd.read_csv(annotation_csv, index_col=0)
    annotation.index = annotation.index.astype(str)
    if cell_type_col not in annotation:
        raise ValueError(f"{annotation_csv} does not contain {cell_type_col!r}")

    frame = scores.rename("raw_score").to_frame()
    frame.index = frame.index.astype(str)
    frame = frame.join(annotation[[cell_type_col]], how="inner").dropna()
    positive = frame[cell_type_col].isin(positive_cell_types)
    n_positive = int(positive.sum())
    n_reference = int((~positive).sum())
    if n_positive < min_cells or n_reference < min_cells:
        raise ValueError(
            "Insufficient annotated cells for score orientation: "
            f"positive={n_positive}, reference={n_reference}, min_cells={min_cells}"
        )

    reducer = np.mean if statistic == "mean" else np.median
    positive_value = float(reducer(frame.loc[positive, "raw_score"]))
    reference_value = float(reducer(frame.loc[~positive, "raw_score"]))
    difference = positive_value - reference_value
    if not np.isfinite(difference) or difference == 0:
        raise ValueError(f"Cannot determine score orientation from difference={difference}")

    multiplier = 1 if difference > 0 else -1
    oriented = scores.astype(float) * multiplier
    diagnostics = {
        "method": "positive_cell_type_contrast",
        "cell_type_col": cell_type_col,
        "positive_cell_types": list(positive_cell_types),
        "statistic": statistic,
        "n_positive": n_positive,
        "n_reference": n_reference,
        "raw_positive_value": positive_value,
        "raw_reference_value": reference_value,
        "raw_positive_minus_reference": difference,
        "multiplier": multiplier,
        "oriented_positive_minus_reference": difference * multiplier,
    }
    return oriented, diagnostics

