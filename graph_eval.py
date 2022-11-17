from pathlib import Path
from typing import Optional, List
from enum import Enum
import time

import typer
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborSampler
import numpy as np
import pandas as pd

from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.utils.utils import set_seed
from happy.graph.visualise import visualize_points
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
)


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"
    intersection = "intersection"


def main(
    seed: int = 0,
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    val_patch_files: Optional[List[str]] = None,
    k: int = 5,
    group_knts: bool = True,
    graph_method: MethodArg = MethodArg.k,
    remove_unlabelled: bool = True,
    annot_tsv: Optional[str] = None,
    verbose: bool = True,
):
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    patch_files = [project_dir / "graph_splits" / file for file in val_patch_files]

    print("Begin graph construction...")
    predictions, embeddings, coords, confidence = get_raw_data(
        project_dir, run_id, x_min, y_min, width, height, verbose=verbose
    )
    # Get ground truth manually annotated data
    _, _, tissue_class = get_groundtruth_patch(
        organ,
        project_dir,
        x_min,
        y_min,
        width,
        height,
        annot_tsv,
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

    data = setup_node_splits(
        data, tissue_class, remove_unlabelled, True, patch_files, verbose=verbose
    )
    print("Graph construction complete")

    # Setup trained model
    pretrained_path = (
        project_dir / "results" / "graph" / exp_name / model_weights_dir / model_name
    )
    model = torch.load(pretrained_path, map_location=device)
    model_epochs = (
        "model_final"
        if model_name == "graph_model.pt"
        else f"model_{model_name.split('_')[0]}"
    )

    # Setup paths
    save_path = (
        Path(*pretrained_path.parts[:-1])
        / "cell_infer"
        / model_epochs
        / f"run_{run_id}"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    plot_name = f"{val_patch_files[0].split('.csv')[0]}"

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
    tissue_class = tissue_class[val_nodes] if annot_tsv is not None else tissue_class

    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled and annot_tsv is not None:
        unlabelled_inds, tissue_class, predicted_labels, pos, out = _remove_unlabelled(
            tissue_class, predicted_labels, pos, out
        )

    # Evaluate against ground truth tissue annotations
    if annot_tsv is not None:
        evaluate(tissue_class, predicted_labels, out, organ, remove_unlabelled)
        evaluation_plots(tissue_class, predicted_labels, organ, save_path)

    # Visualise cluster labels on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / f"{plot_name.split('.png')[0]}.png",
        pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    # make tsv if the whole graph was used
    if len(data.pos) == len(data.pos[data.val_mask]):
        label_dict = {tissue.id: tissue.label for tissue in organ.tissues}
        predicted_labels = [label_dict[label] for label in predicted_labels]
        _save_tissue_preds_as_tsv(predicted_labels, pos, save_path)


def _remove_unlabelled(tissue_class, predicted_labels, pos, out):
    labelled_inds = tissue_class.nonzero()[0]
    tissue_class = tissue_class[labelled_inds]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return labelled_inds, tissue_class, predicted_labels, pos, out


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
