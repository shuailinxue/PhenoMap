import numpy as np
import pandas as pd
import torch


def build_cell_mask(cell_ids, annotation_path=None, cfg=None):
    cfg = cfg or {}
    if not cfg.get("enabled", False):
        return None
    if not annotation_path:
        raise ValueError("cell_filter.enabled=true requires data.annotation_csv")

    anno_df = pd.read_csv(annotation_path, index_col=0).reindex(cell_ids)
    mask = pd.Series(True, index=anno_df.index)

    cell_type_col = cfg.get("cell_type_col", "cell_type")
    contains = cfg.get("cell_type_contains")
    if contains:
        mask &= anno_df[cell_type_col].astype(str).str.contains(contains, case=False, na=False)

    score_col = cfg.get("score_col")
    score_contains = cfg.get("score_col_contains")
    numeric_cols = anno_df.select_dtypes(include=["number"]).columns.tolist()
    if score_col is None and score_contains:
        matches = [c for c in numeric_cols if score_contains in c]
        if not matches:
            raise ValueError(f"No numeric annotation column contains: {score_contains}")
        score_col = matches[0]

    if score_col:
        score = anno_df[score_col].fillna(0)
        mask &= score >= cfg.get("min_score", -np.inf)
        margin = cfg.get("min_margin")
        if margin is not None:
            other_cols = [c for c in numeric_cols if c != score_col]
            max_other = anno_df[other_cols].max(axis=1) if other_cols else 0
            mask &= (score - max_other) >= margin

    return torch.tensor(np.where(mask.fillna(False).values)[0], dtype=torch.long)
