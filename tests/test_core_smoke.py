from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

from step1.data import FullGraphPredictDataset, SpotGraphDataset
from step2.data import sample_subgraph
from step2.models import PhenoMapPhenotypeModel
from step2.utils.score_orientation import orient_scores
from step3.models import PhenoMapModel


def test_step1_dataset_dimensions():
    graph = Data(
        x=torch.ones((3, 4)),
        x_context=torch.ones((3, 4)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    dataset = SpotGraphDataset([graph], torch.ones((1, 2)))
    assert dataset.feat_dim == 4
    assert dataset.gene_dim == 2
    prediction = FullGraphPredictDataset([graph], [["a", "b", "c"]])
    assert prediction.get_cell_ids(0) == ["a", "b", "c"]


def test_spatial_block_sampling_preserves_local_edges():
    n = 100
    pos = torch.column_stack((torch.arange(n, dtype=torch.float32), torch.zeros(n)))
    pairs = []
    for idx in range(n - 1):
        pairs.extend([(idx, idx + 1), (idx + 1, idx)])
    graph = Data(
        x=torch.ones((n, 2)),
        pos=pos,
        edge_index=torch.tensor(pairs, dtype=torch.long).T,
        cell_ids=np.arange(n),
    )
    spatial = sample_subgraph(graph, 20, seed=42, strategy="spatial_block")
    random = sample_subgraph(graph, 20, seed=42, strategy="random")
    assert spatial.edge_index.shape[1] >= 36
    assert spatial.edge_index.shape[1] > random.edge_index.shape[1]


def test_score_orientation_is_global_sign_flip(tmp_path: Path):
    annotation = pd.DataFrame(
        {"cell_type": ["positive", "positive", "other", "other"]},
        index=["1", "2", "3", "4"],
    )
    path = tmp_path / "annotation.csv"
    annotation.to_csv(path)
    raw = pd.Series([-3.0, -2.0, 1.0, 2.0], index=[1, 2, 3, 4])
    oriented, diagnostics = orient_scores(
        raw,
        path,
        cell_type_col="cell_type",
        positive_cell_types=["positive"],
        min_cells=2,
    )
    assert diagnostics["multiplier"] == -1
    np.testing.assert_allclose(oriented.to_numpy(), -raw.to_numpy())


def test_step2_model_forward_shapes():
    model = PhenoMapPhenotypeModel(
        in_features=6, align_dim=4, gnn_dim=4, heads=2, task_type="cox"
    )
    cell_expression = torch.randn(5, 6)
    bulk_expression = torch.randn(3, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    sample_score, embedding, *_ = model(
        cell_expression, edge_index, bulk_expression
    )
    assert sample_score.shape == (3, 1)
    assert embedding.shape == (5, 4)


def test_step3_adapter_forward_shapes():
    config = {
        "model": {
            "keep_vis_dim": 8,
            "keep_text_dim": 8,
            "scgpt_dim": 4,
            "spatial_hidden_dim": 4,
            "spatial_fusion_layers": 2,
            "alignment_mapper_layers": 2,
            "disable_text_similarity": False,
        },
        "loss": {
            "lambda_align": 0.1,
            "lambda_ce": 1.0,
            "scale_factor": 10.0,
            "temperature": 1.0,
        },
    }
    model = PhenoMapModel(config)
    logits, loss, losses = model(
        f_vis=torch.randn(3, 1, 8),
        coords=torch.randn(3, 1, 2),
        e_scgpt=torch.randn(3, 1, 4),
        v_high=torch.randn(1, 8),
        v_low=torch.randn(1, 8),
        hazard_scores=torch.randn(3, 1),
    )
    assert logits.shape == (3, 1, 2)
    assert loss.ndim == 0
    assert set(losses) == {"loss_total", "loss_align", "loss_ce"}


def test_templates_parse():
    for path in Path(".").glob("step*/configs/*.yaml"):
        with path.open(encoding="utf-8") as handle:
            assert isinstance(yaml.safe_load(handle), dict)
