from typing import Optional, List
import json

import typer
from tqdm import tqdm
from torch_geometric.data import Batch

import happy.db.eval_runs_interface as db
from happy.db.graph_model import GraphModel
from happy.utils.utils import get_device, get_project_dir, set_seed
from happy.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.graph.enums import FeatureArg, MethodArg, SupervisedModelsArg
from happy.graph.graph_creation.get_and_process import get_and_process_raw_data
from happy.graph.graph_creation.create_graph import setup_graph
from happy.graph.graph_creation.node_dataset_splits import setup_splits_by_runid
from happy.graph.runners.base import TrainParams, TrainRunner

def main(
    project_name: str = typer.Option(..., help="Project directory"),
    organ_name: str = typer.Option(..., help="Organ name for cell and tissue types"),
    db_name: str = typer.Option("main.db", help="Database file in happy/db/, or an absolute path to a .db file"),
    exp_name: str = typer.Option(..., help="Experiment name for saving results"),
    run_ids: List[int] = typer.Option(..., help="Run IDs of cell graphs to train on"),
    tissue_label_csv: List[str] = typer.Option([], help="Ground truth CSV files, one per run_id from qupath, give either full absolutepath or file name relative to projects/<project_name>/results/tissue_annots/"),
    layers: int = typer.Option(16, help="Number of layers in the model"),
    k: int = typer.Option(5, help="k for kNN graph edge construction"),
    feature: FeatureArg = typer.Option(FeatureArg.embeddings, help="Node features: (.embeddings) or cell (.predictions)"),
    graph_method: MethodArg = typer.Option(MethodArg.intersection, help="Graph edge construction method: kNN (k), Delaunay triangulation (delaunay), or intersecting (intersection)"),
    standardise: bool = True,
    group_knts: bool = typer.Option(True, help="Group KNT predictions into single nodes (placenta-specific)"),
    random_remove: float = typer.Option(0.0, help="Proportion of nodes to randomly remove"),
    top_conf: bool = typer.Option(False, help="Filter to top-confidence cell predictions only"),
    model_type: SupervisedModelsArg = SupervisedModelsArg.sup_clustergcn,
    pretrained: Optional[str] = typer.Option(None, help="Relative path to pretrained model weights"),
    epochs: int = 100,
    include_validation: bool = True,
    validation_step: int = typer.Option(1, help="Run validation every N epochs"),
    patience: int = typer.Option(20, help="Early stopping patience (validation steps without improvement)"),
    hidden_units: int = 256,
    batch_size: int = 200,
    num_neighbours: int = 400,
    dropout: float = 0.5,
    node_dropout: float = 0.0,
    learning_rate: float = 0.001,
    weighted_loss: bool = True,
    use_custom_weights: bool = True,
    val_patch_files: List[str] = typer.Option([], help="CSV files defining validation patch regions"),
    test_patch_files: List[str] = typer.Option([], help="CSV files defining test patch regions"),
    num_workers: int = 7,
    get_cuda_device_num: bool = typer.Option(False, help="Auto-select best GPU"),
    add_to_db: bool = typer.Option(False, help="Register the final trained model in the database"),
    seed: int = 0,
    custom_embeddings_path: Optional[str] = typer.Option(None, help="Custom root path to project directory (overrides default embeddings path)"),
):
    """Train a GNN for tissue node classification on a cell graph.

    Third stage of the HAPPY pipeline (nuc_train > cell_train > tissue_train).
    Takes cell-level embeddings and predictions produced by cell_infer, constructs
    a graph over detected cells, and trains a GNN to assign a tissue type to each cell node.

    Multiple run_ids can be provided; their graphs are batched together for training.
    Results are saved under projects/{project_name}/results/graph/{model_type}/{exp_name}/{timestamp}/.

    either: 
    - split nodes randomly if no val and test patch files are provided (0.15 val, 0.15 test, 0.7 train) #TODO: make these proportions configurable, currently just changed in node dataset splits
    - or give val as bounding box #TODO work out how to not just have as rectangle

    nb in organ make sure unlabelled is class 0

    expected tissue csv to be saved under projects/<project_name>/results/tissue_annots/ , otherwise give full absolute path

    Args:
        project_name: name of the project directory (must exist under projects/)
        organ_name: organ to use for cell and tissue class definitions
        db_name: database filename in happy/db/ (default main.db), or an absolute path to a .db file
        exp_name: experiment name for saving results
        run_ids: list of cell inference run IDs to include in training
        layers: number of GNN layers
        k: number of nearest neighbours for kNN graph construction
        feature: node feature type — 'embeddings' or 'predictions' 
        graph_method: edge construction method — 'k', 'delaunay', or 'intersection'
        standardise: z-score standardise cell embedding features
        group_knts: group KNT cell cluster predictions into single nodes (placenta-specific)
        random_remove: proportion of nodes to randomly drop before graph construction
        top_conf: keep only cells with confidence >= 0.9
        model_type: GNN architecture to use
        pretrained: path to pretrained weights relative to project dir, optional
        epochs: number of training epochs
        hidden_units: hidden units per GNN layer
        batch_size: nodes per training batch
        num_neighbours: neighbours sampled per node (varies by model)
        dropout: model dropout rate
        node_dropout: fraction of nodes to drop per training step
        learning_rate: Adam optimiser learning rate
        weighted_loss: use class-weighted cross entropy loss
        use_custom_weights: use manually tuned class weights
        include_validation: run validation during training
        validation_step: validate every N epochs
        tissue_label_csv: ground truth tissue annotation CSV files, one per run_id
        val_patch_files: CSV files defining spatial regions for validation nodes
        test_patch_files: CSV files defining spatial regions for test nodes
        num_workers: dataloader worker processes
        get_cuda_device_num: auto-select GPU by free memory
        add_to_db: register the final trained model in the database as a GraphModel,
            printing the model id to use with tissue_inference --tissue-model-id
        seed: random seed for reproducibility
    """
    db.init(db_name)
    device = get_device(get_cuda_device_num)
    set_seed(seed)

    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    model_type_val = model_type.value
    feature_val = feature.value
    graph_method_val = graph_method.value

    graph_split_files_dir = project_dir / "graph_splits"
    # empty lists if no val/test patch provided
    resolved_val_patches = (
        [graph_split_files_dir / f for f in val_patch_files] if val_patch_files else []
    )
    resolved_test_patches = (
        [graph_split_files_dir / f for f in test_patch_files] if test_patch_files else []
    )

    # save graph parameters 
    graph_params = {
        "run_ids": run_ids,
        "edge_method": graph_method_val,
        "k": k,
        "feature": feature_val,
        "group_knts": group_knts,
        "random_remove": random_remove,
        "top_conf": top_conf,
        "standardise": standardise,
    }

    # set up logger for training metrics
    logger = Logger(
        ["train", "train_inf", "val"], ["loss", "accuracy"], file=True
    )

    datas = []
    mismatched = []
    # loop over run ids to create a graph for each, then batch together for training
    for i, run_id in enumerate(run_ids):
        # if multiple tissue label CSVs provided, use the corresponding one for each run_id
        csv = tissue_label_csv[i] if i < len(tissue_label_csv) else None
        hdf5_data, tissue_class = get_and_process_raw_data(
            project_name, organ, project_dir, run_id, graph_params, csv,
            custom_path=custom_embeddings_path,
        )
        # create graph for each run_id with tissue class labels
        data = setup_graph(hdf5_data, organ, feature_val, k, graph_method_val, tissue_class)
        # The tissue labels must be one-per-node: if a slide's annotation CSV has a
        # different number of rows than the graph has cells, PyG's ClusterData will
        # silently refuse to partition `y` (is_node_attr only treats tensors whose
        # length == num_nodes as node attributes). The whole batch's `y` then stays
        # full size while train_mask is partitioned, crashing later as a cryptic
        # `mask [..] does not match indexed tensor [..]` during training.
        # Report every slide's node vs label counts before raising, so all bad
        # slides are visible at once rather than failing on the first one.
        n_labels = data.y.size(0) if getattr(data, "y", None) is not None else 0
        is_mismatch = bool(n_labels) and n_labels != data.num_nodes
        print(
            f"run_id {run_id}: {data.num_nodes} cells / nodes, {n_labels} tissue labels"
            + ("  <-- mistmatch" if is_mismatch else "")
        )
        if is_mismatch:
            mismatched.append((run_id, n_labels, data.num_nodes))
        # mask nodes based on provided validation/test patch files
        data = setup_splits_by_runid(
            data,
            tissue_class,
            include_validation,
            resolved_val_patches,
            resolved_test_patches,
        )
        datas.append(data)

    if mismatched:
        details = "; ".join(
            f"run_id {rid}: {nl} labels vs {nn} nodes" for rid, nl, nn in mismatched
        )
        raise ValueError(
            f"Tissue labels do not match the number of graph nodes for "
            f"{len(mismatched)} slide(s) [{details}]"
        )

    # batch multiple subgraphs into one combined graph for training
    #  (so training run over all run_ids in each epoch)
    # node indicies re-indexed:
    # makes disconnected subgraphs to allow batch proccessing
    data = Batch.from_data_list(datas)

    run_params = TrainParams(
        data,
        device,
        pretrained_path,
        model_type_val,
        batch_size,
        num_neighbours,
        epochs,
        layers,
        hidden_units,
        dropout,
        node_dropout,
        learning_rate,
        num_workers,
        weighted_loss,
        use_custom_weights,
        validation_step,
        organ,
    )
    # initialise training runner
    train_runner = TrainRunner.new(run_params)

    # set up directory for saving results and checkpoints, and save graph parameters
    run_path = setup_run(project_dir, f"{model_type_val}/{exp_name}", "graph")
    _save_graph_params(graph_params, run_path)
    train_runner.params.save(seed, exp_name, run_path)

    # train model and save checkpoints !
    best_val = _train(train_runner, logger, run_path, include_validation, patience)

    # optionally register the final model in the database for use by tissue_inference
    if add_to_db:
        final_model_path = run_path / "final_graph_model.pt"
        graph_model = GraphModel.create(
            path=str(final_model_path.resolve()),
            hyperparameters_path=str((run_path / "train_params.json").resolve()),
            exp_name=exp_name,
            model_type=model_type_val,
            organ=organ_name,
            performance=round(best_val, 4),
        )
        print(f"Saved final model to db with id {graph_model.id}")


def _train(train_runner, logger, run_path, include_validation, patience=20):
    """Supervised training loop for tissue node classification.

    Saves the best model checkpoint (by validation accuracy) and a final checkpoint.
    Stops early if validation accuracy does not improve for `patience` validation steps.
    """
    train_runner.prepare_data()
    epochs = train_runner.params.epochs
    validation_step = train_runner.params.validation_step

    prev_best_val = 0
    epochs_without_improvement = 0
    val_accuracy = 0.0

    try:
        pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
        for epoch in pbar:
            loss, accuracy = train_runner.train()
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            if include_validation and (epoch % validation_step == 0 or epoch == 1):
                train_accuracy, val_accuracy = train_runner.validate()
                logger.log_accuracy("train_inf", epoch - 1, train_accuracy)
                logger.log_accuracy("val", epoch - 1, val_accuracy)
                logger.to_csv(run_path / "tissue_train_stats.csv")

                if val_accuracy >= prev_best_val:
                    train_runner.save_state(run_path, epoch)
                    epochs_without_improvement = 0
                    prev_best_val = val_accuracy
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    tqdm.write(f"Early stopping at epoch {epoch} (no improvement for {patience} validation steps)")
                    break

            pbar.set_postfix(
                loss=f"{loss:.4f}",
                train_acc=f"{accuracy:.4f}",
                val_acc=f"{val_accuracy:.4f}",
                best_val=f"{prev_best_val:.4f}",
            )

    except KeyboardInterrupt:
        save = input("Would you like to save anyway? y/n: ")
        if save == "y":
            train_runner.save_state(run_path, epoch)

    train_runner.save_state(run_path, "final")
    tqdm.write(f"Saved final model to {run_path / 'final_graph_model.pt'}")

    return prev_best_val


def _save_graph_params(graph_params_dict, run_path):
    with open(run_path / "graph_params.json", "w") as f:
        json.dump(graph_params_dict, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
