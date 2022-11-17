from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from torch_geometric.data import Data

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.utils.hdf5 import get_datasets_in_patch, get_embeddings_file
from happy.utils.utils import get_project_dir
from happy.graph.visualise import visualize_points, calc_figsize
from happy.graph.create_graph import (
    make_k_graph,
    make_delaunay_triangulation,
    make_intersection_graph,
)


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"
    intersection = "intersection"
    all = "all"


def main(
    run_id: int = typer.Option(...),
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    method: MethodArg = MethodArg.all,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 6,
    plot_edges: bool = False,
):
    """Generates a graph and saves it's visualisation. Node are coloured by cell type

    Args:
        run_id: id of the run which generated the embeddings file
        project_name: name of the project
        organ_name: name of the organ to get the cell colours
        method: graph creation method to use.
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        k: number of neighbours to use for KNN or intersection graph
        plot_edges: whether to plot edges or just points
    """
    # Create database connection
    db.init()

    organ = get_organ(organ_name)
    project_dir = get_project_dir(project_name)

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_dir, run_id)
    print(f"Getting data from: {embeddings_path}")

    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )
    print(f"Data loaded with {len(predictions)} nodes")

    # Make graph data object
    data = Data(x=predictions, pos=torch.Tensor(coords.astype("int32")))

    save_dirs = Path(*embeddings_path.parts[-3:-1])
    save_dir = (
        project_dir / "visualisations" / "graphs" / save_dirs / f"w{width}_h{height}"
    )
    plot_name = f"x{x_min}_y{y_min}"

    method = method.value
    if method == "k":
        vis_for_range_k(k, data, plot_name, save_dir, organ, width, height, plot_edges)
    elif method == "delaunay":
        vis_delaunay(data, plot_name, save_dir, organ, width, height)
    elif method == "intersection":
        vis_intersection(data, k, plot_name, save_dir, organ, width, height)
    elif method == "all":
        vis_for_range_k(k, data, plot_name, save_dir, organ, width, height, plot_edges)
        vis_delaunay(data, plot_name, save_dir, organ, width, height)
        vis_intersection(data, k, plot_name, save_dir, organ, width, height)
    else:
        raise ValueError(f"no such method: {method}")


def vis_for_range_k(
    k, data, plot_name, save_dir, organ, width, height, plot_edges=True
):
    # Specify save graph vis location
    save_path = save_dir / "max_radius"
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate vis for k
    data = make_k_graph(data, k)
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


def vis_delaunay(data, plot_name, save_dir, organ, width, height):
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in data.x]

    # Specify save graph vis location
    save_path = save_dir / "delaunay"
    save_path.mkdir(parents=True, exist_ok=True)

    delaunay = make_delaunay_triangulation(data)
    print(f"Plotting...")

    point_size = 1 if len(delaunay.edges) >= 10000 else 2

    figsize = calc_figsize(data.pos, width, height)
    fig = plt.figure(figsize=figsize, dpi=300)
    plt.triplot(delaunay, linewidth=0.5, color="black")
    plt.scatter(
        data.pos[:, 0], data.pos[:, 1], marker=".", s=point_size, zorder=1000, c=colours
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()

    plot_name = f"{plot_name}.png"
    plt.savefig(save_path / plot_name)
    print(f"Plot saved to {save_path / plot_name}")


def vis_intersection(data, k, plot_name, save_dir, organ, width, height):
    # Specify save graph vis location
    save_path = save_dir / "intersection"
    save_path.mkdir(parents=True, exist_ok=True)

    intersection_graph = make_intersection_graph(data, k)
    edge_index = intersection_graph.edge_index
    edge_weight = intersection_graph.edge_attr

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
