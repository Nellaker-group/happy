import time

import typer
from scipy.special import softmax
import pandas as pd
import numpy as np

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.graph.graph_creation.get_and_process import get_hdf5_data, standardise_cells
from happy.utils.utils import set_seed
from happy.graph.enums import FeatureArg, SupervisedModelsArg, MethodArg
from happy.graph.utils.visualise_points import visualize_points
from happy.graph.utils.utils import get_model_eval_path
from happy.graph.graph_creation.node_dataset_splits import setup_node_splits
from happy.graph.graph_creation.create_graph import setup_graph
from happy.graph.runners.eval_runner import EvalParams, EvalRunner
from happy.hdf5 import get_embeddings_file
from projects.placenta.graphs.processing.process_knots import process_knts


def main(
    seed: int = 0,
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    k: int = 8,
    feature: FeatureArg = FeatureArg.embeddings,
    standardise: bool = True,
    model_type: SupervisedModelsArg = SupervisedModelsArg.sup_clustergcn,
    graph_method: MethodArg = MethodArg.k,
    save_tsv: bool = True,
    save_embeddings: bool = True,
):
    """ Runs inference over a WSI for plotting and saving embeddings and predictions.

    Args:
        seed: set the random seed for reproducibility
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells and tissues
        exp_name: name of the experiment directory to get the model weights from
        model_weights_dir: timestamp directory containing model weights
        model_name: name of the pickle file containing the model weights
        run_id: run_id cells to evaluate over
        k: value of k for kNN graph edge construction method
        feature: one of 'embeddings' or 'predictions'
        standardise: whether to standardise the embeddings
        model_type: which type of supervised graph model the weights are from
        graph_method: which type of edge construction to use
        save_tsv: whether to save the predictions as a tsv file
        save_embeddings: whether to save the embeddings and predictions as a hdf5 file
    """
    db.init()
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    patch_files = [project_dir / "graph_splits" / "all_wsi.csv"]

    hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1)
    # Covert isolated knts into syn and turn groups into a single knt point
    hdf5_data, _ = process_knts(organ, hdf5_data)
    # standardise cell features
    if standardise:
        hdf5_data = standardise_cells(hdf5_data)
    # Covert input cell data into a graph
    data = setup_graph(hdf5_data, organ, feature, k, graph_method)
    # Split graph into an inference set across the WSI
    data = setup_node_splits(data, None, False, True, patch_files)

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
    plot_name = "all_wsi"

    # Visualise cluster labels on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / f"{plot_name.split('.png')[0]}.png",
        data.pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    # Convert outputs to confidence scores
    tissue_confidence = softmax(out, axis=-1)
    top_confidence = tissue_confidence[
        (range(len(predicted_labels)), [predicted_labels])
    ][0]

    # Save predictions to tsv for loading into QuPath
    if save_tsv:
        _save_tissue_preds_as_tsv(predicted_labels, hdf5_data.coords, save_path, organ)

    # Save processed cell and tissue predictions, coordinates and embeddings
    embeddings_path = get_embeddings_file(project_name, run_id, tissue=True)
    if save_embeddings:
        _save_embeddings_as_hdf5(
            hdf5_data,
            predicted_labels,
            graph_embeddings,
            top_confidence,
            embeddings_path,
        )


def _print_prediction_stats(predicted_labels, tissue_label_mapping):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    unique_labels = []
    for label in unique:
        unique_labels.append(tissue_label_mapping[label])
    unique_counts = dict(zip(unique_labels, counts))
    print(f"Counts per label: {unique_counts}")


def _save_embeddings_as_hdf5(
    hdf5_data, tissue_predictions, tissue_embeddings, tissue_confidence, save_path
):
    print("Saving all tissue predictions and embeddings to hdf5")
    hdf5_data.tissue_predictions = tissue_predictions
    hdf5_data.tissue_embeddings = tissue_embeddings
    hdf5_data.tissue_confidence = tissue_confidence
    hdf5_data.to_path(save_path)


def _save_tissue_preds_as_tsv(predicted_labels, coords, save_path, organ):
    print("Saving all tissue predictions as a tsv")
    label_dict = {tissue.id: tissue.label for tissue in organ.tissues}
    predicted_labels = [label_dict[label] for label in predicted_labels]
    tissue_preds_df = pd.DataFrame(
        {
            "x": coords[:, 0].astype(int),
            "y": coords[:, 1].astype(int),
            "class": predicted_labels,
        }
    )
    tissue_preds_df.to_csv(save_path / "tissue_preds.tsv", sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
