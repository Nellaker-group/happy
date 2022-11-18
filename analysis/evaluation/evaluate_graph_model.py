from pathlib import Path
from typing import List
import time

import typer
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborSampler
import numpy as np

from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.utils.utils import set_seed
from happy.graph.create_graph import (
    get_raw_data,
    setup_graph,
    process_knts,
    get_groundtruth_patch,
)
from happy.graph.graph_supervised import (
    inference,
    setup_node_splits,
    evaluate,
    evaluation_plots,
    MethodArg,
)


def main(
    seed: int = 0,
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    pre_trained_path: str = typer.Option(...),
    run_id: int = typer.Option(...),
    annot_tsv: str = typer.Option(...),
    patch_files: List[str] = typer.Option([]),
    k: int = 5,
    group_knts: bool = True,
    graph_method: MethodArg = MethodArg.intersection,
    verbose: bool = True,
    plot: bool = False,
):
    """Evaluates model performance across validation or test datasets

    seed: random seed to fix
    project_name: name of directory containing the project
    organ_name: name of organ
    pre_trained_path: path relative to project to pretrained model
    run_id: evalrun id of embeddings to evaluate over
    annot_tsv: the name of the annotations file containing ground truth points
    patch_files: the name of the file(s) containing validation or test patches
    k: the value of k to use for the kNN or intersection graph
    group_knts: whether to process KNT predictions
    graph_method: method for constructing the graph (k, delaunay, intersection)
    verbose: whether to print to console graph construction progress
    plot: whether to generate evaluation plots
    """
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    patch_files = [project_dir / "graph_splits" / file for file in patch_files]

    print("Begin graph construction...")
    predictions, embeddings, coords, confidence = get_raw_data(
        project_dir, run_id, 0, 0, -1, -1, verbose=verbose
    )
    # Get ground truth manually annotated data
    _, _, tissue_class = get_groundtruth_patch(
        organ, project_dir, 0, 0, -1, -1, annot_tsv
    )
    # Covert isolated knts into syn and turn groups into a single knt point
    if group_knts:
        predictions, embeddings, coords, confidence, tissue_class = process_knts(
            organ,
            predictions,
            embeddings,
            coords,
            confidence,
            tissue_class,
            verbose=verbose,
        )
    # Covert input cell data into a graph
    data = setup_graph(coords, k, embeddings, graph_method, loop=False, verbose=verbose)
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    pos = data.pos
    x = data.x.to(device)

    data = setup_node_splits(data, tissue_class, True, patch_files, verbose=verbose)
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
        Path(*pretrained_path.parts[:-1]) / "eval" / model_epochs / f"run_{run_id}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataloader for eval, feeds in whole graph
    eval_loader = NeighborSampler(
        data.edge_index, node_idx=None, sizes=[-1], batch_size=512, shuffle=False
    )

    # Run inference and get predicted labels for nodes
    timer_start = time.time()
    out, graph_embeddings, predicted_labels = inference(model, x, eval_loader, device)
    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")

    # restrict to only data in patch_files using val_mask
    val_nodes = data.val_mask
    predicted_labels = predicted_labels[val_nodes]
    out = out[val_nodes]
    pos = pos[val_nodes]
    tissue_class = tissue_class[val_nodes]

    # Remove unlabelled (class 0) ground truth points
    unlabelled_inds, tissue_class, predicted_labels, pos, out = _remove_unlabelled(
        tissue_class, predicted_labels, pos, out
    )

    # Evaluate against ground truth tissue annotations
    evaluate(tissue_class, predicted_labels, out, organ, True)
    if plot:
        evaluation_plots(tissue_class, predicted_labels, organ, save_path)


def _remove_unlabelled(tissue_class, predicted_labels, pos, out):
    labelled_inds = tissue_class.nonzero()[0]
    tissue_class = tissue_class[labelled_inds]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return labelled_inds, tissue_class, predicted_labels, pos, out


if __name__ == "__main__":
    typer.run(main)
