import time

import typer
from scipy.special import softmax
import pandas as pd
import numpy as np

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir, set_seed
from happy.organs import get_organ
from happy.graph.graph_creation.get_and_process import get_hdf5_data, standardise_cells
from happy.graph.enums import FeatureArg, SupervisedModelsArg, MethodArg
from happy.graph.utils.visualise_points import visualize_points
from happy.graph.utils.utils import get_model_eval_path
from happy.graph.graph_creation.node_dataset_splits import setup_node_splits
from happy.graph.graph_creation.create_graph import setup_graph
from happy.graph.runners.eval_runner import EvalParams, EvalRunner


def main(
    seed: int = 0,
    project_name: str = typer.Option(..., help="Project directory"),
    organ_name: str = typer.Option(..., help="Organ name for cell and tissue types"),
    db_name: str = "main.db",
    tissue_model_id: int = typer.Option(..., help="ID of the tissue graph model in the database"),
    run_id: int = typer.Option(..., help="Run ID to run inference on"),
    k: int = 8,
    feature: FeatureArg = FeatureArg.embeddings,
    standardise: bool = True,
    group_knts: bool = typer.Option(True, help="Group KNT predictions into single nodes before inference (placenta-specific)"),
    model_type: SupervisedModelsArg = SupervisedModelsArg.sup_clustergcn,
    graph_method: MethodArg = MethodArg.intersection,
    save_tsv: bool = True,
    save_embeddings: bool = typer.Option(True, help="NB will overwrite existing hdf5 file at path and run_id"),
    inf_patches: bool = False,
    custom_path: str = None,
):
    """Run tissue inference over a whole slide image.

    Third stage of the HAPPY pipeline (cell_infer > tissue_infer).
    Loads cell embeddings from HDF5, constructs a spatial graph, runs the trained
    tissue GNN, and saves predictions as a PNG visualisation, TSV, and
    HDF5 for downstream tasks.

    update 24/04/26:
    model weights path is looked up from the database via tissue_model_id.
    Tissue embeddings save path (and this is logged in the db):
    projects/{project_name}/results/embeddings/{lab_id}/{slide_name}/run_{run_id}_tissues.hdf5

    Args:
        seed: random seed for reproducibility
        project_name: name of the project directory
        organ_name: organ name for cell and tissue class definitions
        db_name: name of the db file, default main.db
        tissue_model_id: tissue graph model ID in db
        run_id: run ID of the cell inference output to process
        k: k for kNN graph edge construction
        feature: node feature type — 'embeddings' or 'predictions'
        standardise: z-score standardise cell embedding features before inference
        group_knts: group KNT cell cluster predictions into single nodes (placenta-specific)
        model_type: GNN architecture the weights were trained with
        graph_method: edge construction method — 'k', 'delaunay', or 'intersection'
        save_tsv: save tissue predictions as a TSV for loading into QuPath
        save_embeddings: save tissue predictions, embeddings and confidence to HDF5
        inf_patches: inference on patches
        custom_path: custom path to cell embeddings HDF5 input directory
    """

    db.init(db_name)
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    # for masking nodes during inference (e.g. to only run inference on certain patches)
    if inf_patches:
        patch_files = [project_dir / "graph_splits" / "all_wsi.csv"]

    # load cell data from HDF5 
    hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1, custom_path=custom_path)

    if organ_name == "placenta":
        if group_knts:
            # placenta-specific preprocessing: group KNT cell clusters into single nodes
            # Covert isolated knts into syn and turn groups into a single knt point
            from projects.placenta.graphs.processing.process_knots import process_knts #TODO: move to happy or more general location? or push with placenta project?
            hdf5_data, _ = process_knts(organ, hdf5_data)

    if standardise:
        hdf5_data = standardise_cells(hdf5_data) # make sure trained model also used standardised data

    data = setup_graph(hdf5_data, organ, feature, k, graph_method)
    # mask notes if patch files provided
    if inf_patches:
        data = setup_node_splits(data, None, False, True, patch_files)

    # get tissue model
    pretrained_path = db.get_tissue_model_path(tissue_model_id)
    model_name = pretrained_path.name
    # set up eval runner with trained model weights
    eval_params = EvalParams(data, device, pretrained_path, model_type, 512, organ) 
    eval_runner = EvalRunner.new(eval_params)

    timer_start = time.time()
    print("Running inference")
    out, graph_embeddings, predicted_labels = eval_runner.inference()
    print(f"Total inference time: {time.time() - timer_start:.4f} s")

    save_path = get_model_eval_path(model_name, pretrained_path, run_id)

    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / "all_wsi.png",
        data.pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    tissue_confidence = softmax(out, axis=-1)
    top_confidence = tissue_confidence[
        (range(len(predicted_labels)), [predicted_labels])
    ][0]

    if save_tsv:
        _save_tissue_preds_as_tsv(predicted_labels, hdf5_data.coords, save_path, organ)

    embeddings_path = db.get_tissue_embeddings_path(run_id, project_dir / "results" / "embeddings")
    if save_embeddings:
        _save_embeddings_as_hdf5(
            hdf5_data,
            predicted_labels,
            graph_embeddings,
            top_confidence,
            embeddings_path,
        )

    db.set_tissue_graph_params(run_id, k, graph_method.value)
    db.set_tissue_model(run_id, tissue_model_id)
    db.mark_tissue_as_done(run_id)


def _save_embeddings_as_hdf5(
    hdf5_data, tissue_predictions, tissue_embeddings, tissue_confidence, save_path
):
    print("Saving tissue predictions and embeddings to hdf5")
    hdf5_data.tissue_predictions = tissue_predictions
    hdf5_data.tissue_embeddings = tissue_embeddings
    hdf5_data.tissue_confidence = tissue_confidence
    hdf5_data.to_path(save_path, overwrite=True)


def _save_tissue_preds_as_tsv(predicted_labels, coords, save_path, organ):
    print("Saving tissue predictions as tsv")
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
