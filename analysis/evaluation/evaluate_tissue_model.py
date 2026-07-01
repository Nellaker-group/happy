from typing import List, Optional
import time

import typer
import numpy as np

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.utils.utils import get_device, get_project_dir, set_seed
from happy.graph.graph_creation.get_and_process import get_and_process_raw_data
from happy.graph.graph_creation.create_graph import setup_graph
from happy.graph.enums import FeatureArg, MethodArg, SupervisedModelsArg
from happy.graph.utils.utils import get_model_eval_path
from happy.graph.utils.visualise_points import visualize_points
from happy.graph.utils.evaluation import evaluate, evaluation_plots
from happy.graph.runners.eval_runner import EvalParams, EvalRunner
from happy.graph.graph_creation.node_dataset_splits import setup_node_splits


def main(
    project_name: str = typer.Option(..., help="Project directory name"),
    organ_name: str = typer.Option(..., help="Organ name for cell and tissue types"),
    db_name: str = typer.Option("main.db", help="Database file in happy/db/, or an absolute path to a .db file"),
    custom_embeddings_path: Optional[str] = typer.Option(None, help="Custom root path to the project embeddings (overrides default)"),
    exp_name: str = typer.Option(..., help="Experiment name of the trained model"),
    model_weights_dir: str = typer.Option(..., help="Timestamp directory containing model weights"),
    model_name: str = typer.Option(..., help="Filename of the model weights"),
    run_id: int = typer.Option(..., help="Run ID to evaluate over"),
    seed: int = 0,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    val_patch_files: List[str] = typer.Option([], help="CSV files defining the evaluation region"),
    k: int = 5,
    feature: FeatureArg = FeatureArg.embeddings,
    group_knts: bool = typer.Option(True, help="Group KNT predictions into single nodes (placenta-specific)"),
    random_remove: float = 0.0,
    top_conf: bool = False,
    standardise: bool = True,
    model_type: SupervisedModelsArg = SupervisedModelsArg.sup_clustergcn,
    graph_method: MethodArg = MethodArg.k,
    plot_umap: bool = True,
    remove_unlabelled: bool = True,
    tissue_label_tsv: Optional[str] = typer.Option(None, help="TSV file with ground truth tissue labels"),
    compress_labels: bool = False,
    verbose: bool = True,
):
    """Evaluate a trained tissue GNN on a WSI or subregion.

    Runs inference with a trained model, then produces:
    - Classification metrics (accuracy, F1, confusion matrix) if ground truth provided
    - UMAP plots of cell and tissue embeddings (if --plot-umap)
    - Spatial visualisation of predicted tissue labels

    Sits alongside evaluate_cell_model.py and evaluate_nuclei_model.py.
    For whole-WSI inference without evaluation metrics, use tissue_infer instead.

    Args:
        project_name: name of the project directory
        organ_name: organ name for cell and tissue class definitions
        exp_name: experiment name of the trained model
        model_weights_dir: timestamp subdirectory containing the model weights file
        model_name: filename of the model weights
        run_id: run ID to evaluate over
        seed: random seed for reproducibility
        x_min: left bound of subregion in pixels (0 = full slide)
        y_min: top bound of subregion in pixels (0 = full slide)
        width: width of subregion in pixels (-1 = full width)
        height: height of subregion in pixels (-1 = full height)
        val_patch_files: CSV files defining the spatial evaluation region
        k: k for kNN graph edge construction
        feature: node feature type — 'embeddings' or 'predictions'
        group_knts: group KNT cell cluster predictions into single nodes (placenta-specific)
        random_remove: proportion of nodes to randomly drop before graph construction
        top_conf: keep only cells with confidence >= 0.9
        standardise: z-score standardise cell embedding features
        model_type: GNN architecture the weights were trained with
        graph_method: edge construction method — 'k', 'delaunay', or 'intersection'
        plot_umap: plot UMAP of cell and tissue embeddings (slow for large graphs)
        remove_unlabelled: exclude unlabelled nodes (class 0) from evaluation metrics
        tissue_label_tsv: TSV file with ground truth tissue labels for evaluation metrics
        compress_labels: compress tissue labels into fewer classes (organ-specific)
        verbose: print graph construction details
    """
    db.init(db_name)
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    patch_files = [project_dir / "graph_splits" / file for file in val_patch_files]

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

    hdf5_data, tissue_class = get_and_process_raw_data(
        project_name, organ, project_dir, run_id, graph_params, tissue_label_tsv,
        custom_path=custom_embeddings_path,
    )

    data = setup_graph(hdf5_data, organ, feature, k, graph_method, tissue_class)

    data = setup_node_splits(
        data, tissue_class, remove_unlabelled, True, patch_files, verbose=verbose
    )

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
    print("Running inference")
    out, graph_embeddings, predicted_labels = eval_runner.inference()
    print(f"Total inference time: {time.time() - timer_start:.4f} s")

    save_path = get_model_eval_path(model_name, pretrained_path, run_id)
    conf_str = "_top_conf" if top_conf else ""
    plot_name = (
        f"{val_patch_files[0].split('.csv')[0]}{conf_str}"
        if val_patch_files
        else f"full_wsi{conf_str}"
    )

    val_nodes = data.val_mask
    predicted_labels = predicted_labels[val_nodes]
    out = out[val_nodes]
    pos = data.pos[val_nodes]
    graph_embeddings = graph_embeddings[val_nodes]
    tissue_class = (
        tissue_class[val_nodes] if tissue_label_tsv is not None else tissue_class
    )

    predictions = hdf5_data.cell_predictions
    if remove_unlabelled and tissue_label_tsv is not None:
        unlabelled_inds, tissue_class, predicted_labels, pos, out = _remove_unlabelled(
            tissue_class, predicted_labels, pos, out
        )
        graph_embeddings = graph_embeddings[unlabelled_inds]
        predictions = predictions[unlabelled_inds]

    if plot_umap:
        from happy.graph.embeddings_umap import fit_umap, plot_cell_graph_umap, plot_tissue_umap
        fitted_umap = fit_umap(graph_embeddings)
        plot_cell_graph_umap(
            organ, predictions, fitted_umap, save_path, f"cell_{plot_name}_umap.png"
        )
        plot_tissue_umap(organ, fitted_umap, plot_name, save_path, predicted_labels)
        if tissue_label_tsv is not None:
            plot_tissue_umap(
                organ, fitted_umap, f"gt_{plot_name}", save_path, tissue_class
            )

    tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    _print_prediction_stats(predicted_labels, tissue_label_mapping)

    if tissue_label_tsv is not None:
        _print_prediction_stats(tissue_class, tissue_label_mapping)
        evaluate(tissue_class, predicted_labels, out, organ, compress_labels)
        evaluation_plots(tissue_class, predicted_labels, out, organ, save_path)

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
    unique_labels = [tissue_label_mapping[label] for label in unique]
    unique_counts = dict(zip(unique_labels, counts))
    print(f"Counts per label: {unique_counts}")


if __name__ == "__main__":
    typer.run(main)
