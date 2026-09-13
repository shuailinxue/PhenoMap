"""Reusable analyses for spatial phenotype interpretation and validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, mannwhitneyu, ranksums


def _string_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = result.index.astype(str)
    return result


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Adjust P values with the Benjamini-Hochberg step-up procedure."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    raw = values[valid]
    order = np.argsort(raw)
    ranked = raw[order]
    scaled = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty_like(scaled)
    restored[order] = np.clip(scaled, 0, 1)
    adjusted[valid] = restored
    return adjusted


def define_pr_groups(
    cells: pd.DataFrame,
    *,
    score_col: str = "beta",
    cell_type_col: str = "cell_type",
    malignant_label: str = "Cancer_epithelial",
    high_fraction: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """Assign exactly the highest-scoring fraction of malignant cells to PR-high."""
    if not 0 < high_fraction < 1:
        raise ValueError("high_fraction must be between zero and one")
    result = cells.copy()
    result["pr_group"] = "Not malignant"
    malignant = result[cell_type_col].eq(malignant_label) & result[score_col].notna()
    ordered = result.loc[malignant, score_col].sort_values(
        ascending=False, kind="mergesort"
    )
    if ordered.empty:
        raise ValueError("No scored malignant cells were found")
    n_high = max(1, math.ceil(len(ordered) * high_fraction))
    high_ids = ordered.index[:n_high]
    result.loc[malignant, "pr_group"] = "Remaining 95%"
    result.loc[high_ids, "pr_group"] = "PR-high (top 5%)"
    return result, float(ordered.iloc[n_high - 1])


def load_spatial_expression(
    h5ad_path: Path, beta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cell labels, coordinates, scores, and expression from an AnnData file."""
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    cell_ids = adata.obs_names.astype(str)
    beta = _string_index(beta)
    label_col = "Original_Label" if "Original_Label" in adata.obs else "Ensemble_Label"
    cells = pd.DataFrame(
        {
            "x": np.asarray(adata.obsm["spatial"])[:, 0],
            "y": np.asarray(adata.obsm["spatial"])[:, 1],
            "cell_type": adata.obs[label_col].astype(str).to_numpy(),
        },
        index=cell_ids,
    )
    cells = cells.join(beta[["beta"]], how="inner")
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    expression = pd.DataFrame(matrix, index=cell_ids, columns=adata.var_names.astype(str))
    expression = expression.loc[cells.index]
    return cells, expression


def neighborhood_counts(
    cells: pd.DataFrame,
    *,
    radius: float = 200,
    neighbor_types: Iterable[str] = ("T_cells", "B_cells", "Myeloid", "Stromal"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count specified cell types around each malignant-cell centroid."""
    malignant = cells[cells["pr_group"].isin(["PR-high (top 5%)", "Remaining 95%"])].copy()
    records: list[pd.DataFrame] = []
    tests = []
    for cell_type in neighbor_types:
        neighbors = cells[cells["cell_type"].eq(cell_type)]
        tree = cKDTree(neighbors[["x", "y"]].to_numpy())
        counts = np.fromiter(
            (len(items) for items in tree.query_ball_point(malignant[["x", "y"]], r=radius)),
            dtype=int,
            count=len(malignant),
        )
        frame = pd.DataFrame(
            {
                "center_id": malignant.index,
                "pr_group": malignant["pr_group"].to_numpy(),
                "neighbor_type": cell_type,
                "count": counts,
            }
        )
        records.append(frame)
        high = frame.loc[frame["pr_group"].eq("PR-high (top 5%)"), "count"]
        remaining = frame.loc[frame["pr_group"].eq("Remaining 95%"), "count"]
        statistic, p_value = mannwhitneyu(high, remaining, alternative="two-sided")
        tests.append(
            {
                "cell_type": cell_type,
                "high_mean": high.mean(),
                "remaining_mean": remaining.mean(),
                "u_statistic": statistic,
                "p_value": p_value,
            }
        )
    return pd.concat(records, ignore_index=True), pd.DataFrame(tests)


def marker_neighborhoods(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    *,
    radius: float = 200,
    targets: Iterable[tuple[str, str]] = (
        ("T_cells", "HAVCR2"),
        ("Myeloid", "CD163"),
        ("Stromal", "ACTA2"),
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare marker abundance in cells nearest to PR-high or remaining niches."""
    high = cells[cells["pr_group"].eq("PR-high (top 5%)")][["x", "y"]]
    remaining = cells[cells["pr_group"].eq("Remaining 95%")][["x", "y"]]
    high_tree = cKDTree(high.to_numpy())
    remaining_tree = cKDTree(remaining.to_numpy())
    observations = []
    statistics = []
    for cell_type, gene in targets:
        subset = cells[cells["cell_type"].eq(cell_type)].copy()
        if subset.empty or gene not in expression.columns:
            continue
        coordinates = subset[["x", "y"]].to_numpy()
        high_distance, _ = high_tree.query(coordinates, k=1)
        remaining_distance, _ = remaining_tree.query(coordinates, k=1)
        group = np.full(len(subset), "Outside", dtype=object)
        group[(high_distance <= radius) & (high_distance < remaining_distance)] = "PR-high niche"
        group[(remaining_distance <= radius) & (remaining_distance < high_distance)] = "Remaining niche"
        subset["niche"] = group
        subset["gene"] = gene
        subset["expression"] = expression.loc[subset.index, gene].to_numpy(float)
        subset = subset[subset["niche"].ne("Outside")]
        subset["marker_positive"] = subset["expression"].gt(0)
        observations.append(subset.reset_index(names="cell_id"))

        high_values = subset.loc[subset["niche"].eq("PR-high niche"), "expression"]
        remaining_values = subset.loc[subset["niche"].eq("Remaining niche"), "expression"]
        u_statistic, expression_p = mannwhitneyu(
            high_values, remaining_values, alternative="two-sided"
        )
        table = pd.crosstab(subset["niche"], subset["marker_positive"]).reindex(
            index=["PR-high niche", "Remaining niche"], columns=[False, True], fill_value=0
        )
        _, fraction_p, _, _ = chi2_contingency(table.to_numpy())
        statistics.append(
            {
                "cell_type": cell_type,
                "gene": gene,
                "high_n": len(high_values),
                "remaining_n": len(remaining_values),
                "high_positive_fraction": high_values.gt(0).mean(),
                "remaining_positive_fraction": remaining_values.gt(0).mean(),
                "expression_u": u_statistic,
                "expression_p": expression_p,
                "fraction_chi2_p": fraction_p,
            }
        )
    return pd.concat(observations, ignore_index=True), pd.DataFrame(statistics)


def differential_expression(
    expression: pd.DataFrame, groups: pd.Series
) -> pd.DataFrame:
    """Compute two-sided rank-sum DEA on library-normalized, log-transformed counts."""
    common = expression.index.intersection(groups.index)
    x = expression.loc[common].astype(float)
    group = groups.loc[common]
    library_size = x.sum(axis=1).replace(0, np.nan)
    normalized = x.div(library_size, axis=0).mul(1e4).fillna(0)
    logged = np.log1p(normalized)
    high_mask = group.eq("PR-high (top 5%)").to_numpy()
    rows = []
    for gene in logged.columns:
        statistic, p_value = ranksums(
            logged.loc[high_mask, gene], logged.loc[~high_mask, gene], alternative="two-sided"
        )
        high_mean = normalized.loc[high_mask, gene].mean()
        remaining_mean = normalized.loc[~high_mask, gene].mean()
        rows.append((gene, statistic, np.log2((high_mean + 1e-9) / (remaining_mean + 1e-9)), p_value))
    result = pd.DataFrame(rows, columns=["names", "scores", "logfoldchanges", "pvals"])
    result["pvals_adj"] = benjamini_hochberg(result["pvals"])
    result["minuslog10padj"] = -np.log10(result["pvals_adj"].clip(lower=1e-300))
    result["Significance"] = "Not Sig"
    significant = result["pvals_adj"].lt(0.05)
    result.loc[significant & result["logfoldchanges"].gt(0.5), "Significance"] = "Up"
    result.loc[significant & result["logfoldchanges"].lt(-0.5), "Significance"] = "Down"
    result["abs_score"] = result["scores"].abs()
    return result.sort_values("scores", ascending=False).reset_index(drop=True)


def xgboost_shap_importance(
    expression: pd.DataFrame, groups: pd.Series, *, seed: int = 42
) -> tuple[pd.DataFrame, float]:
    """Fit class-weighted XGBoost and rank genes by mean absolute SHAP value."""
    import shap
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    common = expression.index.intersection(groups.index)
    x = expression.loc[common].astype(float)
    y = groups.loc[common].eq("PR-high (top 5%)").astype(int)
    class_weight = (len(y) - y.sum()) / y.sum()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(class_weight),
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x, y)
    auc = roc_auc_score(y, model.predict_proba(x)[:, 1])
    shap_values = np.asarray(shap.TreeExplainer(model).shap_values(x))
    importance = pd.DataFrame(
        {"gene": x.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return importance, float(auc)


def read_expression_rows(path: Path, genes: Iterable[str], *, chunksize: int = 1000) -> pd.DataFrame:
    """Read selected gene rows from a tab-separated genes-by-samples matrix."""
    wanted = set(map(str, genes))
    pieces = []
    for chunk in pd.read_csv(path, sep="\t", index_col=0, chunksize=chunksize):
        chunk.index = chunk.index.astype(str).str.strip()
        selected = chunk.loc[chunk.index.isin(wanted)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces)
    return result.loc[~result.index.duplicated(keep="first")].apply(pd.to_numeric, errors="coerce")


def _survival_array(clinical: pd.DataFrame, time_col: str, event_col: str) -> np.ndarray:
    return np.array(
        list(zip(clinical[event_col].astype(bool), clinical[time_col].astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def fit_repeated_l1_cox(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    candidates: Iterable[str],
    *,
    time_col: str = "OS.time",
    event_col: str = "OS.event",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a 5-fold, 10-repeat L1-Cox path and apply the one-standard-error rule."""
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    candidate_order = list(dict.fromkeys(map(str, candidates)))[:100]
    expression = _string_index(expression)
    clinical = _string_index(clinical)
    common = expression.columns.astype(str).intersection(clinical.index)
    expression.columns = expression.columns.astype(str)
    c = clinical.loc[common, [time_col, event_col]].apply(pd.to_numeric, errors="coerce").dropna()
    c = c[c[time_col].gt(0) & c[event_col].isin([0, 1])]
    available = [gene for gene in candidate_order if gene in expression.index]
    x = expression.loc[available, c.index].T
    x = x.loc[:, x.notna().all() & x.std().gt(0)]
    z = StandardScaler().fit_transform(x)
    y = _survival_array(c, time_col, event_col)

    full = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=100, alpha_min_ratio=1e-4).fit(z, y)
    alphas = full.alphas_
    splitter = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)
    scores = np.full((50, len(alphas)), np.nan)
    for fold, (train, test) in enumerate(splitter.split(z)):
        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=alphas).fit(z[train], y[train])
        for index, alpha in enumerate(model.alphas_):
            risk = model.predict(z[test], alpha=alpha)
            scores[fold, index] = concordance_index_censored(
                y[test]["event"], y[test]["time"], risk
            )[0]
    mean = np.nanmean(scores, axis=0)
    se = np.nanstd(scores, axis=0, ddof=1) / np.sqrt(np.isfinite(scores).sum(axis=0))
    best = int(np.nanargmax(mean))
    eligible = np.flatnonzero(mean >= mean[best] - se[best])
    one_se = int(eligible[0])
    final = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alphas[one_se]]).fit(z, y)
    coefficients = pd.Series(final.coef_[:, 0], index=x.columns, name="coefficient")
    signature = coefficients[coefficients.gt(1e-10)].sort_values(ascending=False).reset_index()
    signature.columns = ["gene", "coefficient"]
    diagnostics = pd.DataFrame(
        {
            "alpha": alphas,
            "mean_cindex": mean,
            "se": se,
            "best_mean": np.arange(len(alphas)) == best,
            "one_se": np.arange(len(alphas)) == one_se,
        }
    )
    diagnostics.attrs.update(
        candidate_count=len(candidate_order), available_count=x.shape[1], patients=len(c), events=int(c[event_col].sum())
    )
    return signature, diagnostics


def ssgsea_score(expression: pd.DataFrame, genes: Iterable[str]) -> tuple[pd.Series, list[str]]:
    """Compute a single-sample enrichment score for available signature genes."""
    import gseapy as gp

    expression = _string_index(expression)
    available = [gene for gene in dict.fromkeys(map(str, genes)) if gene in expression.index]
    if len(available) < 2:
        raise ValueError("At least two signature genes are required")
    enrichment = gp.ssgsea(
        data=expression,
        gene_sets={"Phenotype signature": available},
        outdir=None,
        no_plot=True,
        permutation_num=0,
        min_size=1,
        max_size=max(500, len(available)),
        threads=1,
        verbose=False,
    ).res2d
    score = enrichment.set_index("Name")["NES"].astype(float)
    score.index = score.index.astype(str)
    return score.rename("score"), available


def evaluate_survival_score(
    score: pd.Series,
    clinical: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    time_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate a z-scored continuous score and a median split."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test

    clinical = _string_index(clinical)
    score = score.copy()
    score.index = score.index.astype(str)
    common = clinical.index.intersection(score.index)
    analysis = clinical.loc[common, [time_col, event_col]].copy()
    analysis["score"] = score.loc[common]
    analysis = analysis.apply(pd.to_numeric, errors="coerce").dropna()
    analysis = analysis[analysis[time_col].gt(0) & analysis[event_col].isin([0, 1])]
    analysis["time_months"] = analysis[time_col] * time_scale
    analysis["score_z"] = (analysis["score"] - analysis["score"].mean()) / analysis["score"].std(ddof=0)
    analysis["group"] = np.where(
        analysis["score_z"].ge(analysis["score_z"].median()), "High", "Low"
    )
    cox_data = analysis[["time_months", event_col, "score_z"]].rename(columns={event_col: "event"})
    cox = CoxPHFitter().fit(cox_data, "time_months", "event")
    interval = np.exp(cox.confidence_intervals_.loc["score_z"].to_numpy())
    high = analysis["group"].eq("High")
    logrank = logrank_test(
        analysis.loc[high, "time_months"],
        analysis.loc[~high, "time_months"],
        analysis.loc[high, event_col],
        analysis.loc[~high, event_col],
    )
    summary = {
        "n": len(analysis),
        "events": int(analysis[event_col].sum()),
        "hr_per_sd": float(cox.hazard_ratios_["score_z"]),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "cox_p": float(cox.summary.loc["score_z", "p"]),
        "logrank_p": float(logrank.p_value),
    }
    return analysis, summary
