"""Deterministic ensemble construction for reference-transfer labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_ensemble_labels(
    tangram_probabilities: pd.DataFrame,
    celltypist_probabilities: pd.DataFrame,
    ingest_labels: pd.Series | pd.DataFrame,
    *,
    probability_threshold: float = 0.4,
) -> pd.DataFrame:
    """Combine Tangram, CellTypist and manifold-transfer predictions."""
    if isinstance(ingest_labels, pd.DataFrame):
        if "prediction" in ingest_labels:
            ingest = ingest_labels["prediction"]
        elif ingest_labels.shape[1] == 1:
            ingest = ingest_labels.iloc[:, 0]
        else:
            raise ValueError("ingest_labels must contain one prediction column.")
    else:
        ingest = ingest_labels
    common = tangram_probabilities.index.intersection(celltypist_probabilities.index).intersection(ingest.index)
    tangram = tangram_probabilities.loc[common]
    celltypist = celltypist_probabilities.loc[common]
    ingest = ingest.loc[common]
    result = pd.DataFrame(index=common)
    result["tangram_prediction"] = tangram.idxmax(axis=1)
    result["celltypist_prediction"] = celltypist.idxmax(axis=1)
    result["ingest_prediction"] = ingest
    result["tangram_probability"] = tangram.max(axis=1)
    result["celltypist_probability"] = celltypist.max(axis=1)

    consensus = result["tangram_prediction"].eq(result["celltypist_prediction"]) & result["tangram_prediction"].eq(result["ingest_prediction"])
    tangram_celltypist = result["tangram_prediction"].eq(result["celltypist_prediction"]) & ~consensus
    tangram_ingest = result["tangram_prediction"].eq(result["ingest_prediction"]) & ~consensus
    celltypist_ingest = result["celltypist_prediction"].eq(result["ingest_prediction"]) & ~consensus
    conflict = ~(consensus | tangram_celltypist | tangram_ingest | celltypist_ingest)
    tangram_wins = conflict & result["tangram_probability"].ge(result["celltypist_probability"]) & result["tangram_probability"].ge(probability_threshold)
    celltypist_wins = conflict & result["celltypist_probability"].gt(result["tangram_probability"]) & result["celltypist_probability"].ge(probability_threshold)
    ingest_fallback = conflict & ~(tangram_wins | celltypist_wins)
    conditions = [consensus, tangram_celltypist, tangram_ingest, celltypist_ingest, tangram_wins, celltypist_wins, ingest_fallback]
    result["cell_type"] = np.select(conditions, [result["tangram_prediction"], result["tangram_prediction"], result["tangram_prediction"], result["celltypist_prediction"], result["tangram_prediction"], result["celltypist_prediction"], result["ingest_prediction"]], default="Error")
    result["decision_level"] = np.select(conditions, ["consensus", "majority_tangram_celltypist", "majority_tangram_ingest", "majority_celltypist_ingest", "high_probability_tangram", "high_probability_celltypist", "low_probability_ingest"], default="error")
    return result
