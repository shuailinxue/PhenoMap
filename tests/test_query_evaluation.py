import numpy as np
import pandas as pd

from step3.utils.query_evaluation import (
    aggregate_spatial_blocks,
    binary_query_performance,
    spatial_block_bootstrap,
    top_fraction_composition,
)


def synthetic_cells():
    return pd.DataFrame(
        {
            "cell_type": ["Cancer_epithelial"] * 8 + ["Stromal"] * 8,
            "x": np.tile([10, 20, 110, 120], 4),
            "y": np.repeat([10, 20, 110, 120], 4),
            "PhenoMap": np.linspace(1, 0, 16),
            "KEEP": np.linspace(0.9, 0.1, 16),
            "molecular_pr": np.linspace(2, -2, 16),
        }
    )


def test_top_composition_and_binary_metrics():
    cells = synthetic_cells()
    composition = top_fraction_composition(cells, ["PhenoMap"], fraction=0.25)
    assert composition["n_top_cells"].sum() == 4
    assert composition.iloc[0]["cell_type"] == "Cancer_epithelial"
    metrics, curves = binary_query_performance(cells, ["PhenoMap", "KEEP"])
    assert np.allclose(metrics["roc_auc"], 1.0)
    assert set(curves) == {"PhenoMap", "KEEP"}


def test_spatial_block_bootstrap_uses_block_units():
    cells = synthetic_cells()
    columns = ["molecular_pr", "PhenoMap", "KEEP"]
    blocks = aggregate_spatial_blocks(cells, columns, block_size=100)
    result = spatial_block_bootstrap(
        cells,
        blocks,
        ["PhenoMap", "KEEP"],
        block_size=100,
        n_bootstrap=20,
        seed=3,
    )
    assert set(result["metric"]) == {
        "block_spearman",
        "cell_pearson",
        "extreme_quintile_auc",
    }
    assert result["n_spatial_blocks"].eq(len(blocks)).all()
    assert result["n_bootstrap"].eq(20).all()
