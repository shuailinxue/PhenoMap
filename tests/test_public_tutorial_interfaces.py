from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from phenomap import load_tutorial_data
from phenomap.cell_annotation import (
    annotate_cells,
    build_ensemble_labels,
    select_marker_genes,
)


def test_tutorial_manifest_loading_and_relative_resolution(tmp_path: Path):
    (tmp_path / "README.md").write_text("PhenoMap", encoding="utf-8")
    (tmp_path / "step1").mkdir()
    (tmp_path / "data").mkdir()
    resource = tmp_path / "data" / "expression.csv"
    resource.write_text("cell_id,GENE\ncell-1,1\n", encoding="utf-8")
    manifest = tmp_path / "tutorial.yaml"
    manifest.write_text(
        yaml.safe_dump({"resources": {"sample": {"expression": "data/expression.csv"}}}),
        encoding="utf-8",
    )

    data = load_tutorial_data(manifest, repository_root=tmp_path)
    assert data.keys() == ("sample.expression",)
    assert data.path("sample.expression") == resource.resolve()
    assert data.require("sample.expression") == resource.resolve()


def test_missing_manifest_resource_error_does_not_expose_resolved_path(tmp_path: Path):
    (tmp_path / "README.md").write_text("PhenoMap", encoding="utf-8")
    (tmp_path / "step1").mkdir()
    manifest = tmp_path / "tutorial.yaml"
    manifest.write_text(
        yaml.safe_dump({"resources": {"sample": {"missing": "private/location.csv"}}}),
        encoding="utf-8",
    )
    data = load_tutorial_data(manifest, repository_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="sample.missing") as error:
        data.require("sample.missing")
    assert str(tmp_path) not in str(error.value)


def test_marker_selection_uses_detection_significance_and_specificity():
    marker_table = pd.DataFrame(
        {
            "cluster": ["T cell", "B cell", "B cell", "T cell"],
            "gene": ["CD3D", "MS4A1", "CD3D", "LOW"],
            "avg_logFC": [2.2, 1.8, 0.5, 3.0],
            "p_val_adj": [0.001, 0.002, 0.003, 0.001],
        }
    )
    expression = pd.DataFrame(
        {"CD3D": [1, 2, 0, 1], "MS4A1": [0, 1, 1, 0], "LOW": [0, 0, 0, 0]}
    )
    selected = select_marker_genes(marker_table, expression)
    assert selected == {"B_cells": ["MS4A1"], "T_cells": ["CD3D"]}


def test_cell_annotation_assigns_highest_positive_marker_score(monkeypatch):
    from phenomap.cell_annotation import annotation as module

    class FakeAnnData:
        def __init__(self, frame):
            self.frame = frame
            self.obs = pd.DataFrame(index=frame.index)
            self.raw = None

        def copy(self):
            return self

    class FakePP:
        @staticmethod
        def normalize_total(adata, target_sum):
            return None

        @staticmethod
        def log1p(adata):
            return None

    class FakeTL:
        @staticmethod
        def score_genes(adata, gene_list, score_name, use_raw):
            adata.obs[score_name] = adata.frame[gene_list].mean(axis=1)

    fake_scanpy = type("FakeScanpy", (), {"AnnData": FakeAnnData, "pp": FakePP, "tl": FakeTL})
    monkeypatch.setattr(module, "_scanpy", lambda: fake_scanpy)
    expression = pd.DataFrame(
        {"EPCAM": [3.0, 0.0, 0.0], "CD3D": [0.0, 2.0, 0.0]},
        index=["cell-1", "cell-2", "cell-3"],
    )
    result = annotate_cells(
        expression,
        {"Cancer_epithelial": ["EPCAM"], "T_cells": ["CD3D"]},
    )
    assert result["cell_type"].tolist() == ["Cancer_epithelial", "T_cells", "Unknown"]
    assert {"Cancer_epithelial_score", "T_cells_score"}.issubset(result)


def test_ensemble_label_construction_covers_consensus_and_fallback():
    index = ["cell-1", "cell-2"]
    tangram = pd.DataFrame({"A": [0.9, 0.35], "B": [0.1, 0.34]}, index=index)
    celltypist = pd.DataFrame({"A": [0.8, 0.33], "B": [0.2, 0.36]}, index=index)
    ingest = pd.Series(["A", "B"], index=index)
    result = build_ensemble_labels(tangram, celltypist, ingest)
    assert result.loc["cell-1", "cell_type"] == "A"
    assert result.loc["cell-1", "decision_level"] == "consensus"
    assert result.loc["cell-2", "cell_type"] == "B"


def test_tutorial_helper_api_imports():
    from step1.utils import plotting as step1_plotting
    from step2.utils import clinical_analysis, plotting as step2_plotting
    from step3.utils import plotting as step3_plotting, query_evaluation

    assert callable(step1_plotting.plot_marker_gene_reconstruction)
    assert callable(step2_plotting.plot_spatial_phenotype_scores)
    assert callable(clinical_analysis.load_spatial_expression)
    assert callable(step3_plotting.plot_query_score_map)
    assert callable(query_evaluation.sample_score_maps)
