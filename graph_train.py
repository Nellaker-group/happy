from typing import Optional, List

import typer
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Batch

from happy.utils.utils import get_device
from happy.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.data.setup_dataloader import setup_graph_dataloaders
from happy.utils.utils import get_project_dir, send_graph_to_device, set_seed
from happy.graph import graph_supervised
from happy.graph.graph_supervised import MethodArg
from happy.graph.create_graph import (
    get_raw_data,
    setup_graph,
    get_groundtruth_patch,
    process_knts,
)


def main(
    seed: int = 0,
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    run_ids: List[int] = typer.Option([]),
    k: int = 5,
    group_knts: bool = True,
    pretrained: Optional[str] = None,
    graph_method: MethodArg = MethodArg.intersection,
    batch_size: int = 200,
    num_neighbours: int = 400,
    epochs: int = 2000,
    layers: int = 16,
    hidden_units: int = 256,
    dropout: float = 0.5,
    learning_rate: float = 0.001,
    weighted_loss: bool = True,
    use_custom_weights: bool = True,
    vis: bool = False,
    annot_tsvs: List[str] = typer.Option([]),
    val_patch_files: List[str] = typer.Option([]),
    test_patch_files: Optional[List[str]] = typer.Option([]),
    validation_step: int = 100,
):
    """Train a ClusterGCN model by constructing a graph on the saved cell embeddings.

    seed: random seed to fix
    project_name: name of directory containing the project
    organ_name: name of organ
    exp_name: a name for this training experiment
    run_ids: the evalrun ids of the slides to get the embeddings from
    k: the value of k to use for the kNN or intersection graph
    group_knts: whether to process KNT predictions
    pretrained: path to a pretrained model (optional)
    graph_method: method for constructing the graph (k, delaunay, intersection)
    batch_size: batch size for training
    num_neighbours: max number of subgraph size for clustergcn
    epochs: number of epochs to train for
    layers: number of graph layers
    hidden_units: number of hidden units per layer
    dropout: amount of dropout to apply at each layer
    learning_rate: the learning rate for the optimizer
    weighted_loss: whether to use weighted loss
    use_custom_weights: if using weighted loss, whether to use custom weights
    vis: whether to use visdom for visualisation
    annot_tsvs: the name of the annotations file containing ground truth points
    val_patch_files: the name of the file(s) containing validation patches
    test_patch_files: the name of the file(s) containing test patches
    validation_step: the epoch step size for which to perform validation
    """

    device = get_device()
    graph_method = graph_method.value
    set_seed(seed)

    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    if len(val_patch_files) > 0:
        val_patch_files = [
            project_dir / "graph_splits" / file for file in val_patch_files
        ]
    if len(test_patch_files) > 0:
        test_patch_files = [
            project_dir / "graph_splits" / file for file in test_patch_files
        ]

    # Setup recording of stats per batch and epoch
    logger = Logger(
        list(["train", "train_inf", "val"]), ["loss", "accuracy"], vis=vis, file=True
    )

    datas = []
    for i, run_id in enumerate(run_ids):
        # Get training data from hdf5 files
        predictions, embeddings, coords, confidence = get_raw_data(
            project_dir, run_id, 0, 0, -1, -1
        )
        # Get ground truth manually annotated data
        _, _, tissue_class = get_groundtruth_patch(
            organ, project_dir, 0, 0, -1, -1, annot_tsvs[i]
        )
        # Covert isolated knts into syn and turn groups into a single knt point
        if group_knts:
            predictions, embeddings, coords, confidence, tissue_class = process_knts(
                organ, predictions, embeddings, coords, confidence, tissue_class
            )
        # Covert input cell data into a graph
        data = setup_graph(coords, k, embeddings, graph_method, loop=False)
        data.y = torch.Tensor(tissue_class).type(torch.LongTensor)
        data = ToUndirected()(data)
        data.edge_index, data.edge_attr = add_self_loops(
            data["edge_index"], data["edge_attr"], fill_value="mean"
        )

        # Split nodes into unlabelled, training and validation sets. So far, validation
        # and test sets are only defined for run_id 1. If there is training data in
        # tissue_class for other runs, that data will also be used for training.
        if run_id == 1:
            data = graph_supervised.setup_node_splits(
                data,
                tissue_class,
                True,
                val_patch_files,
                test_patch_files,
            )
        else:
            data = graph_supervised.setup_node_splits(data, tissue_class, True)
        datas.append(data)

    # Combine multiple graphs into a single graph
    data = Batch.from_data_list(datas)

    # Setup the dataloader which minibatches the graph
    train_loader, val_loader = setup_graph_dataloaders(data, batch_size, num_neighbours)

    # Setup the training parameters
    x, _, _ = send_graph_to_device(data, device)

    # Setup the model
    model = graph_supervised.setup_model(
        data,
        device,
        layers,
        len(organ.tissues),
        hidden_units,
        dropout,
        pretrained_path,
    )

    # Setup training parameters
    optimiser, criterion = graph_supervised.setup_training_params(
        model,
        organ,
        learning_rate,
        train_loader,
        device,
        weighted_loss,
        use_custom_weights,
    )

    # Saves each run by its timestamp and record params for the run
    run_path = setup_run(project_dir, exp_name, "graph")
    params = graph_supervised.collect_params(
        seed,
        organ_name,
        exp_name,
        run_ids,
        0,
        0,
        -1,
        -1,
        k,
        graph_method,
        batch_size,
        num_neighbours,
        learning_rate,
        epochs,
        layers,
        weighted_loss,
        use_custom_weights,
    )
    params.to_csv(run_path / "params.csv", index=False)

    # Train!
    try:
        print("Training:")
        prev_best_val = 0
        for epoch in range(1, epochs + 1):
            loss, accuracy = graph_supervised.train(
                model,
                optimiser,
                criterion,
                train_loader,
                device,
            )
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            if epoch % validation_step == 0 or epoch == 1:
                train_accuracy, val_accuracy = graph_supervised.validate(
                    model, data, val_loader, device
                )
                logger.log_accuracy("train_inf", epoch - 1, train_accuracy)
                logger.log_accuracy("val", epoch - 1, val_accuracy)

                # Save new best model
                if val_accuracy >= prev_best_val:
                    graph_supervised.save_state(run_path, logger, model, epoch)
                    print("Saved best model")
                    prev_best_val = val_accuracy

    except KeyboardInterrupt:
        save_hp = input("Would you like to save anyway? y/n: ")
        if save_hp == "y":
            # Save the fully trained model
            graph_supervised.save_state(run_path, logger, model, epoch)

    # Save the fully trained model
    graph_supervised.save_state(run_path, logger, model, "final")
    print("Saved final model")


if __name__ == "__main__":
    typer.run(main)
