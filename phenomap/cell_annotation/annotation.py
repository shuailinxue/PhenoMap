"""Cell annotation and optional reference-transfer workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .markers import filter_available_markers, map_reference_cell_type


def _scanpy():
    try:
        import scanpy as sc
    except ImportError as error:
        raise ImportError(
            "Reference-transfer regeneration requires the optional annotation dependencies. "
            "Install them with: pip install -r requirements-annotation.txt"
        ) from error
    return sc


def _expression_frame(expression: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(expression, index_col=0) if isinstance(expression, (str, Path)) else expression.copy()
    if frame.empty:
        raise ValueError("Expression data must contain cells and genes.")
    frame.columns = frame.columns.astype(str).str.strip().str.upper()
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _gene_symbols(gene_list: str | Path | Sequence[str] | None, fallback: Sequence[str]) -> list[str]:
    if gene_list is None:
        values = fallback
    elif isinstance(gene_list, (str, Path)):
        values = Path(gene_list).read_text(encoding="utf-8").splitlines()
    else:
        values = gene_list
    return [str(value).strip().upper() for value in values if str(value).strip()]


def _prepare_reference(reference_adata, label_column: str):
    if label_column not in reference_adata.obs:
        raise ValueError(f"Reference atlas does not contain label column {label_column!r}.")
    reference = reference_adata.copy()
    reference.obs["cell_type"] = reference.obs[label_column].map(map_reference_cell_type)
    return reference[reference.obs["cell_type"].notna()].copy()


def _symbol_to_reference_gene(reference) -> dict[str, str]:
    if "feature_name" not in reference.var:
        return {}
    symbols = reference.var["feature_name"].astype(str).str.upper()
    return pd.Series(reference.var_names.astype(str), index=symbols).to_dict()


def _query_adata(frame: pd.DataFrame, symbol_to_gene: Mapping[str, str]):
    sc = _scanpy()
    query = sc.AnnData(frame.astype(np.float32))
    query.var_names = [symbol_to_gene.get(gene, gene) for gene in frame.columns]
    query.var_names_make_unique()
    return query


def annotate_cells(
    expression: str | Path | pd.DataFrame,
    marker_genes: Mapping[str, list[str]],
) -> pd.DataFrame:
    """Assign each cell to the class with the highest Scanpy marker score."""
    sc = _scanpy()
    frame = _expression_frame(expression)
    available = filter_available_markers(marker_genes, frame.columns)
    if not available:
        raise ValueError("None of the configured marker genes occur in the expression matrix.")

    adata = sc.AnnData(frame.astype(np.float32))
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.raw = adata.copy()
    sc.pp.log1p(adata)
    score_columns = {cell_type: f"{cell_type}_score" for cell_type in available}
    for cell_type, genes in available.items():
        sc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name=score_columns[cell_type],
            use_raw=True,
        )

    scores = adata.obs[list(score_columns.values())].copy()
    labels = scores.rename(columns={value: key for key, value in score_columns.items()}).idxmax(axis=1)
    labels = labels.where(scores.max(axis=1) > 0, "Unknown")
    result = scores.copy()
    result.insert(0, "cell_type", labels)
    return result


def run_ingest_transfer(
    expression: str | Path | pd.DataFrame,
    reference_adata,
    *,
    label_column: str = "celltype_major",
    gene_list: str | Path | Sequence[str] | None = None,
    n_components: int = 50,
    n_neighbors: int = 15,
) -> pd.DataFrame:
    """Reproduce the accepted PCA/KNN manifold-label transfer."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    sc = _scanpy()
    frame = _expression_frame(expression)
    reference = _prepare_reference(reference_adata, label_column)
    symbol_to_gene = _symbol_to_reference_gene(reference)
    query = _query_adata(frame, symbol_to_gene)
    target_genes = [symbol_to_gene.get(gene, gene) for gene in _gene_symbols(gene_list, frame.columns)]
    reference_genes = set(reference.var_names)
    query_genes = set(query.var_names)
    common = [gene for gene in target_genes if gene in reference_genes and gene in query_genes]
    if not common:
        raise ValueError("The reference atlas and measured expression have no configured genes in common.")
    reference = reference[:, common].copy()
    query = query[:, common].copy()
    sc.pp.normalize_total(query, target_sum=1e4)
    sc.pp.log1p(query)
    reference_values = reference.X.toarray() if hasattr(reference.X, "toarray") else np.asarray(reference.X)
    query_values = query.X.toarray() if hasattr(query.X, "toarray") else np.asarray(query.X)
    reference_values = np.nan_to_num(reference_values, nan=0.0)
    query_values = np.nan_to_num(query_values, nan=0.0)
    components = min(n_components, reference_values.shape[0] - 1, reference_values.shape[1])
    pca = PCA(n_components=components, random_state=42)
    reference_pca = pca.fit_transform(reference_values)
    query_pca = pca.transform(query_values)
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
    classifier.fit(reference_pca, reference.obs["cell_type"])
    return pd.DataFrame({"prediction": classifier.predict(query_pca)}, index=query.obs_names)


def run_celltypist_transfer(
    expression: str | Path | pd.DataFrame,
    reference_adata,
    *,
    label_column: str = "celltype_major",
    gene_list: str | Path | Sequence[str] | None = None,
    n_jobs: int = 10,
) -> pd.DataFrame:
    """Train the accepted seven-class CellTypist model and return query probabilities."""
    try:
        import celltypist
    except ImportError as error:
        raise ImportError(
            "CellTypist transfer requires the optional annotation dependencies. "
            "Install them with: pip install -r requirements-annotation.txt"
        ) from error
    sc = _scanpy()
    from scipy import sparse

    frame = _expression_frame(expression)
    reference = _prepare_reference(reference_adata, label_column)
    symbol_to_gene = _symbol_to_reference_gene(reference)
    target_genes = [
        symbol_to_gene.get(gene) for gene in _gene_symbols(gene_list, frame.columns)
        if symbol_to_gene.get(gene) in reference.var_names
    ] if symbol_to_gene else _gene_symbols(gene_list, frame.columns)
    target_genes = [gene for gene in target_genes if gene in reference.var_names]
    if not target_genes:
        raise ValueError("The reference atlas does not contain the configured annotation genes.")
    if float(reference.X.max()) > 50:
        sc.pp.normalize_total(reference, target_sum=1e4)
        sc.pp.log1p(reference)
    reference = reference[:, target_genes].copy()
    if sparse.issparse(reference.X):
        reference.X.data = np.nan_to_num(reference.X.data, nan=0.0)
    else:
        reference.X = np.nan_to_num(reference.X, nan=0.0)
    model = celltypist.train(
        reference,
        labels="cell_type",
        check_expression=False,
        n_jobs=n_jobs,
    )

    query = _query_adata(frame, symbol_to_gene)
    model_genes = list(model.features)
    aligned = np.zeros((query.n_obs, len(model_genes)), dtype=np.float32)
    query_lookup = {gene: index for index, gene in enumerate(query.var_names)}
    model_lookup = {gene: index for index, gene in enumerate(model_genes)}
    for gene in set(query_lookup).intersection(model_lookup):
        values = query.X[:, query_lookup[gene]]
        aligned[:, model_lookup[gene]] = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
    aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
    aligned[aligned < 0] = 0
    empty = aligned.sum(axis=1) == 0
    if empty.any():
        aligned[empty, 0] = 1e-4
    aligned_query = sc.AnnData(aligned, obs=query.obs.copy())
    aligned_query.var_names = model_genes
    sc.pp.normalize_total(aligned_query, target_sum=1e4)
    sc.pp.log1p(aligned_query)
    aligned_query.X = np.nan_to_num(aligned_query.X, nan=0.0)
    prediction = celltypist.annotate(aligned_query, model=model, majority_voting=False)
    probabilities = prediction.probability_matrix.copy()
    probabilities.index = frame.index
    return probabilities


def run_tangram_transfer(
    expression: str | Path | pd.DataFrame,
    reference_adata,
    *,
    label_column: str = "celltype_major",
    gene_list: str | Path | Sequence[str] | None = None,
    device: str | None = None,
    epochs: int = 500,
    batch_size: int = 20_000,
) -> pd.DataFrame:
    """Run the accepted batched Tangram cell mapping and project class probabilities."""
    try:
        import tangram as tg
        import torch
    except ImportError as error:
        raise ImportError(
            "Tangram transfer requires the optional annotation dependencies. "
            "Install them with: pip install -r requirements-annotation.txt"
        ) from error
    sc = _scanpy()
    frame = _expression_frame(expression)
    reference = _prepare_reference(reference_adata, label_column)
    symbol_to_gene = _symbol_to_reference_gene(reference)
    query = _query_adata(frame, symbol_to_gene)
    target_genes = [symbol_to_gene.get(gene, gene) for gene in _gene_symbols(gene_list, frame.columns)]
    reference_genes = set(reference.var_names)
    query_genes = set(query.var_names)
    common = [gene for gene in target_genes if gene in reference_genes and gene in query_genes]
    if not common:
        raise ValueError("The reference atlas and measured expression have no configured genes in common.")
    sc.pp.normalize_total(query, target_sum=1e4)
    sc.pp.log1p(query)
    query = query[query.X.sum(axis=1).A1 > 0].copy() if hasattr(query.X.sum(axis=1), "A1") else query[np.asarray(query.X.sum(axis=1)).ravel() > 0].copy()
    reference = reference[:, common].copy()
    query = query[:, common].copy()
    categories = sorted(reference.obs["cell_type"].unique())
    class_matrix = pd.get_dummies(reference.obs["cell_type"])[categories].to_numpy()
    selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    chunks = []
    for start in range(0, query.n_obs, batch_size):
        chunk = query[start : start + batch_size].copy()
        tg.pp_adatas(reference, chunk, genes=common)
        mapping = tg.map_cells_to_space(
            reference,
            chunk,
            mode="cells",
            density_prior="uniform",
            num_epochs=epochs,
            device=selected_device,
        )
        matrix = mapping.X.toarray() if hasattr(mapping.X, "toarray") else np.asarray(mapping.X)
        values = matrix.T @ class_matrix
        probabilities = pd.DataFrame(values, index=chunk.obs_names, columns=categories)
        chunks.append(probabilities.div(probabilities.sum(axis=1), axis=0).fillna(0.0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.concat(chunks)
