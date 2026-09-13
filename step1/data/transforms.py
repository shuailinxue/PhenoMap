from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph


def normalize_gene_expression(
    spot_expression_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(spot_expression_df)
    return (
        pd.DataFrame(
            scaled_data,
            index=spot_expression_df.index,
            columns=spot_expression_df.columns,
        ),
        scaler,
    )


def create_cell_spot_mapping(
    cell_data_df: pd.DataFrame,
    spot_geo_df: pd.DataFrame,
    spot_pixel_size: float,
) -> pd.DataFrame:
    spot_centers = spot_geo_df[["spot_center_x", "spot_center_y"]].to_numpy(
        dtype=np.float32
    )
    cell_centers = cell_data_df[["center_x", "center_y"]].to_numpy(dtype=np.float32)
    distances = np.sqrt(
        np.sum(
            (cell_centers[:, np.newaxis, :] - spot_centers[np.newaxis, :, :]) ** 2,
            axis=2,
        )
    )
    cell_indices, spot_indices = np.where(distances <= spot_pixel_size / 2.0)
    return pd.DataFrame(
        {
            "cell_id": cell_data_df["cell_id"].to_numpy()[cell_indices],
            "spot_id": spot_geo_df["spot_id"].to_numpy()[spot_indices],
        }
    )


def create_spot_cell_map(
    features_local_df: pd.DataFrame,
    features_context_df: pd.DataFrame,
    cell_metadata_df: pd.DataFrame,
    spot_expression_df: pd.DataFrame,
    cell_spot_map_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, Dict], Dict[str, np.ndarray]]:
    labels = spot_expression_df.apply(
        lambda row: row.to_numpy(dtype=np.float32), axis=1
    ).to_dict()
    spot_to_cells = cell_spot_map_df.groupby("spot_id")["cell_id"].apply(list).to_dict()
    graph_inputs: Dict[str, Dict] = {}

    for spot_id in spot_expression_df.index:
        cell_ids = spot_to_cells.get(spot_id)
        if not cell_ids:
            continue
        try:
            local = features_local_df.loc[cell_ids].to_numpy(dtype=np.float32)
            context = features_context_df.loc[cell_ids].to_numpy(dtype=np.float32)
            coords = cell_metadata_df.loc[cell_ids, ["center_x", "center_y"]]
        except KeyError:
            continue
        if len(local):
            graph_inputs[spot_id] = {
                "features_local": local,
                "features_context": context,
                "coords": coords,
            }

    spot_ids = [spot_id for spot_id in spot_expression_df.index if spot_id in graph_inputs]
    return spot_ids, graph_inputs, labels


def create_radius_graph(
    features_local: torch.Tensor,
    features_context: torch.Tensor,
    cell_coords: pd.DataFrame,
    radius: float,
) -> Data:
    coords = cell_coords.to_numpy() if isinstance(cell_coords, pd.DataFrame) else cell_coords
    pos = torch.as_tensor(coords, dtype=torch.float32)
    edge_index = radius_graph(pos, r=radius, loop=False, flow="source_to_target")
    return Data(
        x=features_local,
        x_context=features_context,
        edge_index=edge_index,
        pos=pos,
    )


def build_full_prediction_graph(
    features_local: torch.Tensor,
    features_context: torch.Tensor,
    coords: torch.Tensor,
    spot_radius: float,
) -> Data:
    edge_index = radius_graph(
        coords, r=spot_radius, loop=False, flow="source_to_target"
    )
    return Data(
        x=features_local,
        x_context=features_context,
        edge_index=edge_index,
    )

