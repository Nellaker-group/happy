import pandas as pd
from torch_geometric.utils import add_self_loops
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.transforms import Distance, KNNGraph, ToUndirected
from scipy.spatial import Voronoi
from tqdm import tqdm
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from happy.graph.utils.utils import get_feature


def setup_cell_tissue_graph(hdf5_data, k, graph_method):
    # combine cell and tissue embeddings into one feature
    feature_data = np.concatenate(
        (hdf5_data.cell_embeddings, hdf5_data.tissue_embeddings), axis=1
    )
    data = construct_graph(hdf5_data.coords, k, feature_data, graph_method, loop=False)
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    return data


def setup_cell_graph(hdf5_data, k, graph_method):
    feature_data = hdf5_data.cell_embeddings
    data = construct_graph(hdf5_data.coords, k, feature_data, graph_method, loop=False)
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    return data


def setup_graph(hdf5_data, organ, feature, k, graph_method, tissue_class=None):
    feature_data = get_feature(
        feature, hdf5_data.cell_predictions, hdf5_data.cell_embeddings, organ
    )
    data = construct_graph(hdf5_data.coords, k, feature_data, graph_method, loop=False)
    if tissue_class is not None:
        data.y = torch.Tensor(tissue_class).type(torch.LongTensor)
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    return data


def construct_graph(
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
    data = KNNGraph(k=k + 1, loop=loop, force_undirected=False)(data)
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
    knn_graph = KNNGraph(k=k + 1, loop=False, force_undirected=False)(data)
    knn_edge_index = knn_graph.edge_index.T
    knn_edge_index = knn_edge_index.contiguous().numpy()
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


def get_list_of_subgraphs(
    data,
    tile_coordinates,
    tile_width,
    tile_height,
    min_cells_in_tile,
    max_tiles=-1,
    plot_nodes_per_tile=False,
):
    # Create list of tiles which hold their node indicies from the graph
    tiles = []
    for i, tile_coords in enumerate(
        tqdm(tile_coordinates, desc="Extract nodes within tiles")
    ):
        node_inds = get_nodes_within_tiles(
            tile_coords, tile_width, tile_height, data["pos"][:, 0], data["pos"][:, 1]
        )
        tiles.append(
            {
                "tile_index": i,
                "min_x": tile_coords[0],
                "min_y": tile_coords[1],
                "node_inds": node_inds,
                "num_nodes": len(node_inds),
            }
        )
    tiles = pd.DataFrame(tiles)

    # Plot histogram of number of nodes per tile
    if plot_nodes_per_tile:
        _plot_nodes_per_tile(tiles, binwidth=25)

    # Remove tiles with number of cell points below a min threshold
    nodeless_tiles = tiles[tiles.num_nodes < min_cells_in_tile]
    print(f"Removing {len(nodeless_tiles)}/{len(tiles)} tiles with too few nodes")
    tiles.drop(nodeless_tiles.index, inplace=True)
    tiles.reset_index(drop=True, inplace=True)

    # Create a datasets of subgraphs based on the tile nodes
    tiles_node_inds = tiles["node_inds"].to_numpy()
    removed_tiles = list(nodeless_tiles.index)
    return make_tile_graph_dataset(tiles_node_inds, data, max_tiles), removed_tiles


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


def _plot_nodes_per_tile(tiles, binwidth):
    plt.figure(figsize=(8, 8))
    sns.displot(tiles, x="num_nodes", binwidth=binwidth, color="blue")
    plt.savefig("plots/num_nodes_per_tile.png")
    plt.clf()
    plt.close("all")


def make_tile_graph_dataset(tile_nodes, full_graph, max_tiles):
    tile_graphs = []
    for i, node_inds in enumerate(tqdm(tile_nodes, desc="Get subgraphs within tiles")):
        if i > max_tiles:
            break
        tile_edge_index, tile_edge_attr = subgraph(
            node_inds,
            full_graph["edge_index"],
            full_graph["edge_attr"],
            relabel_nodes=True,
        )
        tile_graph = Data(
            x=full_graph["x"][node_inds],
            edge_index=tile_edge_index,
            edge_attr=tile_edge_attr,
            pos=full_graph["pos"][node_inds],
        )
        tile_graphs.append(tile_graph)
    return tile_graphs
