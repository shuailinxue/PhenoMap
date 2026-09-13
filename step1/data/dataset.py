from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class SpotGraphDataset(Dataset):
    def __init__(self, spot_graphs: List[Data], y_labels: torch.Tensor) -> None:
        if len(spot_graphs) != y_labels.shape[0]:
            raise ValueError(
                f"Expected one label per graph, got {len(spot_graphs)} graphs "
                f"and {y_labels.shape[0]} labels"
            )
        self.spot_graphs = spot_graphs
        self.y_labels = y_labels

    @property
    def gene_dim(self) -> int:
        return self.y_labels.shape[1]

    @property
    def feat_dim(self) -> int:
        return self.spot_graphs[0].x.shape[1]

    def __len__(self) -> int:
        return len(self.spot_graphs)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        return self.spot_graphs[idx], self.y_labels[idx]


class FullGraphPredictDataset(Dataset):
    def __init__(self, graphs: List[Data], cell_ids_list: List[List[str]]) -> None:
        if len(graphs) != len(cell_ids_list):
            raise ValueError(
                f"Expected one cell-ID list per graph, got {len(graphs)} graphs "
                f"and {len(cell_ids_list)} lists"
            )
        self.graphs = graphs
        self.cell_ids_list = cell_ids_list

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

    def get_cell_ids(self, idx: int) -> List[str]:
        return self.cell_ids_list[idx]

