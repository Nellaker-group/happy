import os

import pandas as pd
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.transforms import Distance, KNNGraph
from scipy.spatial import Voronoi
import matplotlib.tri as tri
import numpy as np
import torch

from happy.utils.hdf5 import get_embeddings_file, get_datasets_in_patch
from happy.graph.process_knts import process_knt_cells


def get_groundtruth_patch(organ, project_dir, x_min, y_min, width, height, annot_tsv):
    if not annot_tsv:
        print("No tissue annotation tsv supplied")
        return None, None, None
    tissue_label_path = project_dir / "annotations" / "graph" / annot_tsv
    if not os.path.exists(str(tissue_label_path)):
        print("No tissue label tsv found")
        return None, None, None

    ground_truth_df = pd.read_csv(tissue_label_path, sep="\t")
    xs = ground_truth_df["px"].to_numpy()
    ys = ground_truth_df["py"].to_numpy()
    tissue_classes = ground_truth_df["class"].to_numpy()

    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        sort_args = np.lexsort((ys, xs))
        tissue_ids = np.array(
            [organ.tissue_by_label(tissue_name).id for tissue_name in tissue_classes]
        )
        return xs[sort_args], ys[sort_args], tissue_ids[sort_args]

    mask = np.logical_and(
        (np.logical_and(xs > x_min, (ys > y_min))),
        (np.logical_and(xs < (x_min + width), (ys < (y_min + height)))),
    )
    patch_xs = xs[mask]
    patch_ys = ys[mask]
    patch_tissue_classes = tissue_classes[mask]

    patch_tissue_ids = np.array(
        [organ.tissue_by_label(tissue_name).id for tissue_name in patch_tissue_classes]
    )
    sort_args = np.lexsort((patch_ys, patch_xs))

    return patch_xs[sort_args], patch_ys[sort_args], patch_tissue_ids[sort_args]


def get_raw_data(
    project_dir, run_id, x_min, y_min, width, height, verbose=True
):
    embeddings_path = get_embeddings_file(project_dir, run_id)
    if verbose:
        print(f"Getting data from: {embeddings_path}")
        print(f"Using patch of size: x{x_min}, y{y_min}, w{width}, h{height}")
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height, verbose=verbose
    )
    if verbose:
        print(f"Data loaded with {len(predictions)} nodes")
    sort_args = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[sort_args]
    predictions = predictions[sort_args]
    embeddings = embeddings[sort_args]
    confidence = confidence[sort_args]
    if verbose:
        print("Data sorted by x coordinates")

    return predictions, embeddings, coords, confidence


def process_knts(
    organ, predictions, embeddings, coords, confidence, tissues=None, verbose=True
):
    # Turn isolated knts into syn and group large knts into one point
    predictions, embeddings, coords, confidence, inds_to_remove = process_knt_cells(
        predictions,
        embeddings,
        coords,
        confidence,
        organ,
        50,
        3,
        verbose=verbose,
    )
    # Remove points from tissue ground truth as well
    if tissues is not None and len(inds_to_remove) > 0:
        tissues = np.delete(tissues, inds_to_remove, axis=0)
    return predictions, embeddings, coords, confidence, tissues


def setup_graph(
    coords, k, feature, graph_method, norm_edges=True, loop=True, verbose=True
):
    data = Data(x=torch.Tensor(feature), pos=torch.Tensor(coords.astype("int32")))
    if graph_method == "k":
        graph = make_k_graph(data, k, norm_edges, loop, verbose=verbose)
    elif graph_method == "delaunay":
        graph = make_delaunay_graph(data, norm_edges, verbose=verbose)
    elif graph_method == "intersection":
        graph = make_intersection_graph(data, k, norm_edges, verbose=verbose)
    else:
        raise ValueError(f"No such graph method: {graph_method}")
    if graph.x.ndim == 1:
        graph.x = graph.x.view(-1, 1)
    return graph


def make_k_graph(data, k, norm_edges=True, loop=True, verbose=True):
    if verbose:
        print(f"Generating graph for k={k}")
    data = KNNGraph(k=k + 1, loop=loop, force_undirected=True)(data)
    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    if verbose:
        print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def make_radius_k_graph(data, radius, k, verbose=True):
    if verbose:
        print(f"Generating graph for radius={radius} and k={k}")
    data.edge_index = radius_graph(data.pos, r=radius, max_num_neighbors=k)
    if verbose:
        print("Graph made!")
    return data


def make_voronoi(data, verbose=True):
    if verbose:
        print(f"Generating voronoi diagram")
    vor = Voronoi(data.pos)
    if verbose:
        print("Voronoi made!")
    return vor


def make_delaunay_triangulation(data, verbose=True):
    if verbose:
        print(f"Generating delaunay triangulation")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    if verbose:
        print("Triangulation made!")
    return triang


def make_delaunay_graph(data, norm_edges=True, verbose=True):
    if verbose:
        print(f"Generating delaunay graph")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    data.edge_index = torch.tensor(triang.edges.astype("int64"), dtype=torch.long).T
    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    if verbose:
        print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def make_intersection_graph(data, k, norm_edges=True, verbose=True):
    if verbose:
        print(f"Generating graph for k={k}")
    knn_graph = KNNGraph(k=k + 1, loop=False, force_undirected=True)(data)
    knn_edge_index = knn_graph.edge_index.T
    knn_edge_index = np.array(knn_edge_index.tolist())
    if verbose:
        print(f"Generating delaunay graph")
    try:
        triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
        delaunay_edge_index = triang.edges.astype("int64")
    except ValueError:
        print("Too few points to make a triangulation, returning knn graph")
        get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
        knn_graph = get_edge_distance_weights(knn_graph)
        if verbose:
            print(f"Graph made with {len(knn_graph.edge_index[0])} edges!")
        return knn_graph

    if verbose:
        print(f"Generating intersection of both graphs")
    _, ncols = knn_edge_index.shape
    dtype = ", ".join([str(knn_edge_index.dtype)] * ncols)
    intersection = np.intersect1d(
        knn_edge_index.view(dtype), delaunay_edge_index.view(dtype)
    )
    intersection = intersection.view(knn_edge_index.dtype).reshape(-1, ncols)
    intersection = torch.tensor(intersection, dtype=torch.long).T
    data.edge_index = intersection

    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    if verbose:
        print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def get_nodes_within_tiles(tile_coords, tile_width, tile_height, all_xs, all_ys):
    tile_min_x, tile_min_y = tile_coords[0], tile_coords[1]
    tile_max_x, tile_max_y = tile_min_x + tile_width, tile_min_y + tile_height
    if isinstance(all_xs, torch.Tensor) and isinstance(all_ys, torch.Tensor):
        mask = torch.logical_and(
            (torch.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
            (torch.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
        )
        return mask.nonzero()[:, 0].tolist()
    else:
        mask = np.logical_and(
            (np.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
            (np.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
        )
        return mask.nonzero()[0].tolist()
