from typing import Optional, List
import time

import typer
import numpy as np

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.utils.utils import get_device, get_project_dir, set_seed
from happy.graph.embeddings_umap import fit_umap, plot_cell_graph_umap, plot_tissue_umap
from happy.graph.graph_creation.get_and_process import get_and_process_raw_data
from happy.graph.graph_creation.create_graph import setup_graph
from happy.graph.enums import FeatureArg, MethodArg, SupervisedModelsArg
from happy.graph.utils.utils import get_model_eval_path
from happy.graph.utils.visualise_points import visualize_points
from happy.graph.utils.evaluation import evaluate, evaluation_plots
from happy.graph.runners.eval_runner import EvalParams, EvalRunner
from happy.graph.graph_creation.node_dataset_splits import setup_node_splits


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
    feature: FeatureArg = FeatureArg.embeddings,
    group_knts: bool = True,
    random_remove: float = 0.0,
    top_conf: bool = False,
    standardise: bool = True,
    model_type: SupervisedModelsArg = SupervisedModelsArg.sup_clustergcn,
    graph_method: MethodArg = MethodArg.k,
    plot_umap: bool = True,
    remove_unlabelled: bool = True,
    tissue_label_tsv: Optional[str] = None,
    compress_labels: bool = False,
    verbose: bool = True,
):
    """ Runs inference over a WSI or a region of a WSI. If ground truth data is
    provided, it will produce evaluation metrics and plots.

    Args:
        seed: set the random seed for reproducibility
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells and tissues
        exp_name: name of the experiment directory to get the model weights from
        model_weights_dir: timestamp directory containing model weights
        model_name: name of the pickle file containing the model weights
        run_id: run_id cells to evaluate over
        x_min: bottom left x coordinate of the region to evaluate over (0 for full WSI)
        y_min: bottom left y coordinate of the region to evaluate over (0 for full WSI)
        width: width of the region to evaluate over (-1 for full WSI)
        height: height of the region to evaluate over (-1 for full WSI)
        val_patch_files: list of files containing validation or test regions
        k: value of k for kNN graph edge construction method
        feature: one of 'embeddings' or 'predictions'
        group_knts: whether to first group knt predictions into one node
        random_remove: what proportion of the nodes to randomly remove
        top_conf: whether to filter cells by top confidence predictions only
        standardise: whether to standardise the cell embedding input features
        model_type: which type of supervised graph model the weights are from
        graph_method: which type of edge construction to use
        plot_umap: whether to plot the UMAP of the cell and tissue embeddings (slow)
        remove_unlabelled: whether to remove unlabelled nodes from the evaluation
        tissue_label_tsv: tsv file containing ground truth tissue labels for each cell
        compress_labels: whether to compress the labels into fewer classes (in organ)
        verbose: whether to print graph setup
    """
    db.init()
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    patch_files = [project_dir / "graph_splits" / file for file in val_patch_files]

    # Graph params for saving
    graph_params = {
        "run_ids": run_id,
        "x_min": x_min,
        "y_min": y_min,
        "width": width,
        "height": height,
        "edge_method": graph_method,
        "k": k,
        "feature": feature,
        "group_knts": group_knts,
        "random_remove": random_remove,
        "top_conf": top_conf,
        "standardise": standardise,
    }
    # Get and process raw and ground truth data
    hdf5_data, tissue_class = get_and_process_raw_data(
        project_name, organ, project_dir, run_id, graph_params, tissue_label_tsv
    )

    # Covert input cell data into a graph
    data = setup_graph(hdf5_data, organ, feature, k, graph_method, tissue_class)

    # Split data into validation or test set based on val_patch_files
    data = setup_node_splits(
        data, tissue_class, remove_unlabelled, True, patch_files, verbose=verbose
    )

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / model_type.value
        / exp_name
        / model_weights_dir
        / model_name
    )
    eval_params = EvalParams(data, device, pretrained_path, model_type, 512, organ)
    eval_runner = EvalRunner.new(eval_params)

    timer_start = time.time()
    # Run inference and get predicted labels for nodes
    print("Running inference")
    out, graph_embeddings, predicted_labels = eval_runner.inference()
    timer_end = time.time()
    print(f"total inference time: {timer_end - timer_start:.4f} s")

    # Setup path to save results
    save_path = get_model_eval_path(model_name, pretrained_path, run_id)
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"{val_patch_files[0].split('.csv')[0]}{conf_str}"

    # restrict to only data in patch_files using val_mask
    val_nodes = data.val_mask
    predicted_labels = predicted_labels[val_nodes]
    out = out[val_nodes]
    pos = data.pos[val_nodes]
    graph_embeddings = graph_embeddings[val_nodes]
    tissue_class = (
        tissue_class[val_nodes] if tissue_label_tsv is not None else tissue_class
    )

    predictions = hdf5_data.cell_predictions
    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled and tissue_label_tsv is not None:
        unlabelled_inds, tissue_class, predicted_labels, pos, out = _remove_unlabelled(
            tissue_class, predicted_labels, pos, out
        )
        graph_embeddings = graph_embeddings[unlabelled_inds]
        predictions = predictions[unlabelled_inds]

    if plot_umap:
        # fit and plot umap with cell classes
        fitted_umap = fit_umap(graph_embeddings)
        plot_cell_graph_umap(
            organ, predictions, fitted_umap, save_path, f"cell_{plot_name}_umap.png"
        )
        # Plot the predicted labels onto the umap of the graph embeddings
        plot_tissue_umap(organ, fitted_umap, plot_name, save_path, predicted_labels)
        if tissue_label_tsv is not None:
            plot_tissue_umap(
                organ, fitted_umap, f"gt_{plot_name}", save_path, tissue_class
            )

    # Print some prediction count info
    tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    _print_prediction_stats(predicted_labels, tissue_label_mapping)

    # Evaluate against ground truth tissue annotations
    if tissue_label_tsv is not None:
        _print_prediction_stats(tissue_class, tissue_label_mapping)
        evaluate(tissue_class, predicted_labels, out, organ, compress_labels)
        evaluation_plots(tissue_class, predicted_labels, out, organ, save_path)

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


def _remove_unlabelled(tissue_class, predicted_labels, pos, out):
    labelled_inds = tissue_class.nonzero()[0]
    tissue_class = tissue_class[labelled_inds]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return labelled_inds, tissue_class, predicted_labels, pos, out


def _print_prediction_stats(predicted_labels, tissue_label_mapping):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    unique_labels = []
    for label in unique:
        unique_labels.append(tissue_label_mapping[label])
    unique_counts = dict(zip(unique_labels, counts))
    print(f"Counts per label: {unique_counts}")


if __name__ == "__main__":
    typer.run(main)
