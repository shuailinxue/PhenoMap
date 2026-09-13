import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.ndimage import center_of_mass
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph


def load_target_roi_graph(
    sc_expr_path,
    mask_path,
    common_genes,
    roi_limits=None,
    spot_radius=50.0,
    random_downsample=False,
    downsample_frac=0.2,
    seed=42,
):
    sc_df = pd.read_csv(sc_expr_path, index_col=0)
    sc_df = sc_df.reindex(columns=common_genes).fillna(0)

    mask = tifffile.imread(mask_path)
    query_ids = sc_df.index.astype(int).values
    centers_yx = np.array(center_of_mass(mask, labels=mask, index=query_ids))
    coords_df = pd.DataFrame(centers_yx, index=sc_df.index, columns=["y", "x"]).dropna()
    sc_df = sc_df.loc[coords_df.index]

    if roi_limits:
        roi_mask = (
            (coords_df["x"] >= roi_limits["x_min"])
            & (coords_df["x"] <= roi_limits["x_max"])
            & (coords_df["y"] >= roi_limits["y_min"])
            & (coords_df["y"] <= roi_limits["y_max"])
        )
        coords_df = coords_df[roi_mask]
        sc_df = sc_df.loc[coords_df.index]
        print(f"Retained {len(sc_df)} cells")

    if random_downsample:
        sc_df = sc_df.sample(frac=downsample_frac, random_state=seed)
        coords_df = coords_df.loc[sc_df.index]
        print(f"Retained {len(sc_df)} cells after downsampling")

    x = torch.FloatTensor(sc_df.values)
    pos = torch.FloatTensor(coords_df.values)
    edge_index = radius_graph(pos, r=spot_radius, loop=False)
    return Data(x=x, edge_index=edge_index, pos=pos, cell_ids=sc_df.index.values)


def load_full_graph(
    sc_expr_path,
    mask_path,
    common_genes,
    spot_radius=50.0,
    seed=42,
):
    print("Loading full expression matrix...")
    sc_df = pd.read_csv(sc_expr_path, index_col=0)
    sc_df = sc_df.reindex(columns=common_genes).fillna(0)
    print(f"  {len(sc_df)} cells x {len(common_genes)} genes")

    print("Extracting cell centroids from mask...")
    mask = tifffile.imread(mask_path)
    query_ids = sc_df.index.astype(int).values
    centers_yx = np.array(center_of_mass(mask, labels=mask, index=query_ids))
    coords_df = pd.DataFrame(centers_yx, index=sc_df.index, columns=["y", "x"]).dropna()
    sc_df = sc_df.loc[coords_df.index]
    print(f"  {len(sc_df)} cells with valid coordinates")

    x = torch.FloatTensor(sc_df.values)
    pos = torch.FloatTensor(coords_df.values)

    print(f"Building radius graph (r={spot_radius})...")
    edge_index = radius_graph(pos, r=spot_radius, loop=False)
    print(f"  {edge_index.shape[1]} edges")

    return Data(x=x, edge_index=edge_index, pos=pos, cell_ids=sc_df.index.values)


def sample_subgraph(data, n_nodes, seed=None, strategy="spatial_block"):
    total = data.x.shape[0]
    n = min(n_nodes, total)
    generator = torch.Generator(device=data.x.device)
    if seed is not None:
        generator.manual_seed(seed)
    if strategy == "spatial_block":
        center_idx = torch.randint(
            total, (1,), generator=generator, device=data.x.device
        ).item()
        distance = torch.sum((data.pos - data.pos[center_idx]) ** 2, dim=1)
        idx = torch.topk(distance, k=n, largest=False).indices
    elif strategy == "random":
        idx = torch.randperm(total, generator=generator, device=data.x.device)[:n]
    else:
        raise ValueError("subgraph strategy must be 'spatial_block' or 'random'")
    idx_sorted, _ = idx.sort()

    node_mask = torch.zeros(total, dtype=torch.bool, device=data.x.device)
    node_mask[idx_sorted] = True
    src, dst = data.edge_index
    edge_mask = node_mask[src] & node_mask[dst]
    sub_edge = data.edge_index[:, edge_mask]

    new_id = torch.full((total,), -1, dtype=torch.long, device=data.x.device)
    new_id[idx_sorted] = torch.arange(n, device=data.x.device)
    sub_edge = new_id[sub_edge]

    return Data(
        x=data.x[idx_sorted],
        edge_index=sub_edge,
        pos=data.pos[idx_sorted],
        cell_ids=data.cell_ids[idx_sorted.cpu().numpy()],
    )
