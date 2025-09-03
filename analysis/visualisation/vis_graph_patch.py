from enum import Enum
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from matplotlib.collections import LineCollection
from torch_geometric.data import Data

import happy.db.eval_runs_interface as db
from happy.graph.utils.visualise_points import visualize_points
from happy.organs import get_organ
from happy.hdf5 import get_embeddings_file
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from projects.placenta.graphs.processing.process_knots import process_knts
from happy.utils.utils import get_project_dir
from happy.graph.graph_creation.create_graph import (
    make_k_graph,
    make_radius_k_graph,
    make_voronoi,
    make_delaunay_graph,
    make_intersection_graph,
)


class MethodArg(str, Enum):
    k = "k"
    radius = "radius"
    voronoi = "voronoi"
    delaunay = "delaunay"
    intersection = "intersection"
    all = "all"


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    organ_name: str = "placenta",
    method: MethodArg = MethodArg.all,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    top_conf: bool = False,
    plot_edges: bool = False,
    group_knts: bool = False,
    single_cell: Optional[str] = None,
    percent_to_keep: float = 0.0,
    custom_save_dir: Optional[str] = None,
):
    """Generates a graph and saves its visualisation. Node are coloured by cell type

    Args:
        run_id: id of the run which generated the embeddings file
        project_name: name of the project
        organ_name: name of the organ to get the cell colours
        method: graph creation method to use.
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        top_conf: filter the nodes to only those >90% network confidence
        plot_edges: whether to plot edges or just points
        group_knts: whether to group knts into a single node
        single_cell: filter the nodes to only those of a single cell type
        percent_to_keep: percent of nodes to random remove down to
        custom_save_dir: custom directory to save the graph visualisation
    """
    # Create database connection
    db.init()

    organ = get_organ(organ_name)
    project_dir = get_project_dir(project_name)

    # Get hdf5 datasets contained in specified box/patch of WSI
    embeddings_path = get_embeddings_file(project_name, run_id)
    hdf5_data = get_hdf5_data(project_name, run_id, x_min, y_min, width, height)

    if group_knts:
        hdf5_data, _ = process_knts(organ, hdf5_data)
    if top_conf:
        hdf5_data = hdf5_data.filter_by_confidence(0.9, 1.0)
    if single_cell:
        hdf5_data = hdf5_data.filter_by_cell_type(single_cell)

    # Make graph data object
    data = Data(
        x=torch.Tensor(hdf5_data.cell_predictions),
        pos=torch.Tensor(hdf5_data.coords.astype("int32")),
    )

    keep_indices = None
    if percent_to_keep > 0.0:
        num_to_keep = int(data.num_nodes * percent_to_keep)
        keep_indices = torch.LongTensor(
            np.random.choice(np.arange(data.num_nodes), num_to_keep, replace=False)
        )

    # setup save location and filename
    save_dirs = Path(*embeddings_path.parts[-3:-1])
    save_dirs = custom_save_dir / save_dirs if custom_save_dir else save_dirs
    save_dir = (
        project_dir / "visualisations" / "graphs" / save_dirs / f"w{width}_h{height}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"x{x_min}_y{y_min}_top_conf" if top_conf else f"x{x_min}_y{y_min}"
    plot_name = f"{plot_name}_{single_cell}" if single_cell else plot_name
    plot_name = (
        f"{plot_name}_{percent_to_keep}_reduced" if percent_to_keep > 0.0 else plot_name
    )

    method = method.value
    if method == "k":
        vis_for_range_k(
            6,
            7,
            data,
            plot_name,
            save_dir,
            organ,
            width,
            height,
            plot_edges,
            keep_indices,
        )
    elif method == "radius":
        vis_for_range_radius(
            200, 260, 20, data, plot_name, save_dir, organ, keep_indices
        )
    elif method == "voronoi":
        vis_voronoi(data, plot_name, save_dir, organ)
    elif method == "delaunay":
        vis_delaunay(data, plot_name, save_dir, organ, width, height, keep_indices)
    elif method == "intersection":
        vis_intersection(
            data, 6, plot_name, save_dir, organ, width, height, keep_indices
        )
    elif method == "all":
        vis_for_range_k(
            6,
            7,
            data,
            plot_name,
            save_dir,
            organ,
            width,
            height,
            plot_edges,
            keep_indices,
        )
        vis_voronoi(data, plot_name, save_dir, organ)
        vis_delaunay(data, plot_name, save_dir, organ, width, height, keep_indices)
    else:
        raise ValueError(f"no such method: {method}")


def vis_for_range_k(
    k_start,
    k_end,
    data,
    plot_name,
    save_dir,
    organ,
    width,
    height,
    plot_edges=True,
    keep_indices=None,
):
    # Specify save graph vis location
    save_path = save_dir / "max_radius"
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate vis for different values of k
    for k in range(k_start, k_end):
        data = make_k_graph(data, k)
        if keep_indices is not None:
            data = data.subgraph(keep_indices)
        if not plot_edges:
            edge_index = None
            edge_weight = None
        else:
            edge_index = data.edge_index
            edge_weight = data.edge_attr

        plot_name = f"k{k}_{plot_name}.png"
        print(f"Plotting...")
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            labels=data.x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            width=width,
            height=height,
        )
        print(f"Plot saved to {save_path / plot_name}")


def vis_for_range_radius(
    rad_start, rad_end, k, data, plot_name, save_dir, organ, keep_indices=None
):
    for radius in range(rad_start, rad_end, 10):
        # Specify save graph vis location
        save_path = save_dir / f"radius_{radius}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate vis for radius and k
        data = make_radius_k_graph(data, radius, k)
        if keep_indices is not None:
            data = data.subgraph(keep_indices)

        print(f"Plotting...")
        plot_name = f"k{k}_{plot_name}.png"
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            labels=data.x,
            edge_index=data.edge_index,
        )
        print(f"Plot saved to {save_path / plot_name}")


def vis_voronoi(data, plot_name, save_dir, organ, show_points=False):
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in data.x]

    # Specify save graph vis location
    save_path = save_dir / "voronoi"
    save_path.mkdir(parents=True, exist_ok=True)

    vor = make_voronoi(data)
    print(f"Plotting...")

    point_size = 0.5 if len(vor.vertices) >= 10000 else 1

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.gca()
    finite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            vertices = vor.vertices[simplex]
            if (
                vertices[:, 0].min() >= vor.min_bound[0]
                and vertices[:, 0].max() <= vor.max_bound[0]
                and vertices[:, 1].min() >= vor.min_bound[1]
                and vertices[:, 1].max() <= vor.max_bound[1]
            ):
                finite_segments.append(vertices)
    if show_points:
        plt.scatter(
            vor.points[:, 0],
            vor.points[:, 1],
            marker=".",
            s=point_size,
            zorder=1000,
            c=colours,
        )
    else:
        xys = np.array(finite_segments).reshape(-1, 2)
        plt.scatter(xys[:, 0], xys[:, 1], marker=".", s=point_size, zorder=1000)
    ax.add_collection(
        LineCollection(
            finite_segments, colors="black", lw=0.5, alpha=0.6, linestyle="solid"
        )
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()

    plot_name = f"{plot_name}.png"
    plt.savefig(save_path / plot_name)
    print(f"Plot saved to {save_path / plot_name}")


def vis_delaunay(data, plot_name, save_dir, organ, width, height, keep_indices=None):
    # Specify save graph vis location
    save_path = save_dir / "delaunay"
    save_path.mkdir(parents=True, exist_ok=True)

    data = make_delaunay_graph(data)
    if keep_indices is not None:
        data = data.subgraph(keep_indices)
    edge_index = data.edge_index
    edge_weight = data.edge_attr

    print(f"Plotting...")
    visualize_points(
        organ,
        save_path / plot_name,
        data.pos,
        labels=data.x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        width=width,
        height=height,
    )
    print(f"Plot saved to {save_path / plot_name}")


def vis_intersection(
    data, k, plot_name, save_dir, organ, width, height, keep_indices=None
):
    # Specify save graph vis location
    save_path = save_dir / "intersection"
    save_path.mkdir(parents=True, exist_ok=True)

    data = make_intersection_graph(data, k)
    if keep_indices is not None:
        data = data.subgraph(keep_indices)
    edge_index = data.edge_index
    edge_weight = data.edge_attr

    plot_name = f"k{k}_{plot_name}.png"
    print(f"Plotting...")
    visualize_points(
        organ,
        save_path / plot_name,
        data.pos,
        labels=data.x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        width=width,
        height=height,
    )
    print(f"Plot saved to {save_path / plot_name}")


if __name__ == "__main__":
    typer.run(main)
