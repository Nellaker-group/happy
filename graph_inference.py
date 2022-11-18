from pathlib import Path
import time

import typer
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborSampler
import pandas as pd

from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.utils.utils import set_seed
from happy.graph.graph_supervised import inference, MethodArg
from happy.graph.visualise import visualize_points
from happy.graph.create_graph import (
    get_raw_data,
    setup_graph,
    process_knts,
)


def main(
    seed: int = 0,
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    pre_trained_path: str = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 5,
    group_knts: bool = True,
    graph_method: MethodArg = MethodArg.intersection,
    verbose: bool = True,
):
    """Generates a visualisation of the graph model predictions for the specified
    region. Will save a tsv of all predictions if the whole slide is used, which
    can be loaded into QuPath.

    seed: random seed to fix
    project_name: name of directory containing the project
    organ_name: name of organ
    pre_trained_path: path relative to project to pretrained model
    x_min: the top left x coordinate of the patch to use
    y_min: the top left y coordinate of the patch to use
    width: the width of the patch to use. -1 for all
    height: the height of the patch to use. -1 for all
    k: the value of k to use for the kNN or intersection graph
    group_knts: whether to process KNT predictions
    graph_method: method for constructing the graph (k, delaunay, intersection)
    verbose: whether to print to console graph construction progress
    """
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    print("Begin graph construction...")
    predictions, embeddings, coords, confidence = get_raw_data(
        project_dir, run_id, x_min, y_min, width, height, verbose=verbose
    )

    # Covert isolated knts into syn and turn groups into a single knt point
    if group_knts:
        predictions, embeddings, coords, confidence, _ = process_knts(
            organ,
            predictions,
            embeddings,
            coords,
            confidence,
            verbose=verbose,
        )
    # Covert input cell data into a graph
    data = setup_graph(
        coords, k, embeddings, graph_method, loop=False, verbose=verbose
    )
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    pos = data.pos
    x = data.x.to(device)
    print("Graph construction complete")

    # Setup trained model
    pretrained_path = project_dir / pre_trained_path
    model = torch.load(pretrained_path, map_location=device)
    model_name = pretrained_path.parts[-1]
    model_epochs = (
        "model_final"
        if model_name == "graph_model.pt"
        else f"model_{model_name.split('_')[0]}"
    )

    # Setup paths
    save_path = (
        Path(*pretrained_path.parts[:-1])
        / "eval"
        / model_epochs
        / f"run_{run_id}"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}.png"

    # Dataloader for eval, feeds in whole graph
    eval_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=[-1],
        batch_size=512,
        shuffle=False,
    )

    # Run inference and get predicted labels for nodes
    timer_start = time.time()
    out, graph_embeddings, predicted_labels = inference(
        model, x, eval_loader, device
    )
    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")

    # Visualise cluster labels on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / plot_name,
        pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    # make tsv if the whole graph was used
    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        label_dict = {tissue.id: tissue.label for tissue in organ.tissues}
        predicted_labels = [label_dict[label] for label in predicted_labels]
        _save_tissue_preds_as_tsv(predicted_labels, pos, save_path)


def _save_tissue_preds_as_tsv(predicted_labels, coords, save_path):
    print("Saving all tissue predictions as a tsv")
    tissue_preds_df = pd.DataFrame(
        {
            "x": coords[:, 0].numpy().astype(int),
            "y": coords[:, 1].numpy().astype(int),
            "class": predicted_labels,
        }
    )
    tissue_preds_df.to_csv(save_path / "tissue_preds.tsv", sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
