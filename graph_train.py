from typing import Optional, List
import json

import typer
from torch_geometric.data import Batch

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device
from happy.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.utils.utils import get_project_dir, set_seed
from happy.graph.enums import FeatureArg, MethodArg, SupervisedModelsArg
from happy.graph.graph_creation.get_and_process import get_and_process_raw_data
from happy.graph.graph_creation.create_graph import setup_graph
from happy.graph.runners.train_runner import TrainParams, TrainRunner
from happy.graph.graph_creation.node_dataset_splits import setup_splits_by_runid


def main(
    seed: int = 0,
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    run_ids: List[int] = typer.Option([]),
    k: int = 5,
    feature: FeatureArg = FeatureArg.embeddings,
    group_knts: bool = True,
    random_remove: float = 0.0,
    pretrained: Optional[str] = None,
    top_conf: bool = False,
    model_type: SupervisedModelsArg = SupervisedModelsArg.sup_clustergcn,
    graph_method: MethodArg = MethodArg.k,
    batch_size: int = 200,
    num_neighbours: int = 400,
    epochs: int = 100,
    layers: int = typer.Option(...),
    hidden_units: int = 256,
    dropout: float = 0.5,
    node_dropout: float = 0.0,
    learning_rate: float = 0.001,
    standardise: bool = True,
    weighted_loss: bool = True,
    use_custom_weights: bool = True,
    num_workers: int = 12,
    vis: bool = False,
    tissue_label_tsvs: List[str] = typer.Option([]),
    val_patch_files: Optional[List[str]] = [],
    test_patch_files: Optional[List[str]] = [],
    include_validation: bool = True,
    validation_step: int = 50,
):
    """ Train a supervised model on a cell graph for tissue node classification.

    Args:
        seed: set the random seed for reproducibility
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells and tissues
        exp_name: name of the experiment directory to save results to
        run_ids: the run_ids of the cell graphs to use for training
        k: value of k for kNN graph edge construction method
        feature: one of 'embeddings' or 'predictions'
        group_knts: whether to first group knt predictions into one node
        random_remove: what proportion of the nodes to randomly remove
        pretrained: path to pretraind model (optional)
        top_conf: whether to filter cells by top confidence predictions only
        model_type: which type of supervised graph model to use - default clusterGCN
        graph_method: which type of edge construction to use
        batch_size: number of nodes per batch
        num_neighbours: varies by model_type, broadly the sampled neighbours per node
        epochs: number of epochs to train for
        layers: number of layers in the model
        hidden_units: number of hidden units per layer in the model
        dropout: model parameter dropout rate
        node_dropout: node dropout rate
        learning_rate: learning rate for the optimizer
        standardise: whether to standardise the cell embedding input features
        weighted_loss: whether to use weighted loss (usually true)
        use_custom_weights: whether to use custom weights
        num_workers: number of workers for the dataloader
        vis: whether to use visdom for visualisation
        tissue_label_tsvs: list of tsv files containing tissue labels for each run_id
        val_patch_files: list of files with regions to use for validation
        test_patch_files: list of files with regions to use for test
        include_validation: whether to include validation in training
        validation_step: at how many epochs to run validation
    """
    # general setup
    db.init()
    device = get_device()
    set_seed(seed)

    model_type = model_type.value
    graph_method = graph_method.value
    feature = feature.value

    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    graph_split_files_dir = project_dir / "graph_splits"
    if len(val_patch_files) > 0:
        val_patch_files = [graph_split_files_dir / file for file in val_patch_files]
    if len(test_patch_files) > 0:
        test_patch_files = [graph_split_files_dir / file for file in test_patch_files]

    # Graph params for saving
    graph_params = {
        "run_ids": run_ids,
        "edge_method": graph_method,
        "k": k,
        "feature": feature,
        "group_knts": group_knts,
        "random_remove": random_remove,
        "top_conf": top_conf,
        "standardise": standardise,
    }

    # Setup recording of stats per batch and epoch
    logger = Logger(
        list(["train", "train_inf", "val"]), ["loss", "accuracy"], vis=vis, file=True
    )

    datas = []
    for i, run_id in enumerate(run_ids):
        # Get and process raw and ground truth data
        hdf5_data, tissue_class = get_and_process_raw_data(
            project_name, organ, project_dir, run_id, graph_params, tissue_label_tsvs[i]
        )

        # Covert input cell data into a graph
        data = setup_graph(hdf5_data, organ, feature, k, graph_method, tissue_class)

        # Split data into train, val, test sets based on patch files and run_id
        data = setup_splits_by_runid(
            data,
            run_id,
            tissue_class,
            include_validation,
            val_patch_files,
            test_patch_files,
        )
        datas.append(data)

    # Combine multiple graphs into a single graph
    data = Batch.from_data_list(datas)

    # Setup training parameters, including dataloaders, models, loss, etc.
    run_params = TrainParams(
        data,
        device,
        pretrained_path,
        model_type,
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
    train_runner = TrainRunner.new(run_params)

    # Saves each run by its timestamp and record params for the run
    run_path = setup_run(project_dir, f"{model_type}/{exp_name}", "graph")
    _save_graph_params(graph_params, run_path)
    train_runner.params.save(seed, exp_name, run_path)

    # train!
    train(train_runner, logger, run_path, include_validation)


def train(train_runner, logger, run_path, include_validation):
    train_runner.prepare_data()
    epochs = train_runner.params.epochs
    validation_step = train_runner.params.validation_step

    try:
        print("Training:")
        prev_best_val = 0
        for epoch in range(1, epochs + 1):
            loss, accuracy = train_runner.train()
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            if include_validation and (epoch % validation_step == 0 or epoch == 1):
                train_accuracy, val_accuracy = train_runner.validate()
                logger.log_accuracy("train_inf", epoch - 1, train_accuracy)
                logger.log_accuracy("val", epoch - 1, val_accuracy)
                logger.to_csv(run_path / "graph_train_stats.csv")
                # Save new best model
                if val_accuracy >= prev_best_val:
                    train_runner.save_state(run_path, epoch)
                    print("Saved best model")
                    prev_best_val = val_accuracy

    except KeyboardInterrupt:
        save_hp = input("Would you like to save anyway? y/n: ")
        if save_hp == "y":
            # Save the interrupted model
            train_runner.save_state(run_path, epoch)

    # Save the final trained model
    train_runner.save_state(run_path, "final")
    print("Saved final model")


def _save_graph_params(graph_params_dict, run_path):
    with open(run_path / "graph_params.json", "w") as f:
        json.dump(graph_params_dict, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
