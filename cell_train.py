from typing import List, Optional
import os

import typer

from happy.train.hyperparameters import Hyperparameters
from happy.utils.utils import get_device, get_project_dir
from happy.logger.logger import Logger
from happy.organs import get_organ
from happy.train import cell_train, utils


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    model_name: str = "resnet-50",
    pre_trained: Optional[str] = None,
    num_workers: int = 16,
    epochs: int = 20,
    batch: int = 600,
    val_batch: int = 600,
    learning_rate: float = 1e-4,
    decay_gamma: float = 1,
    step_size: int = 20,
    init_from_inc: bool = False,
    frozen: bool = True,
    weighted_loss: bool = False,
    vis: bool = True,
    get_cuda_device_num: bool = False,
):
    """For training a cell classification model

    Multiple datasets can be combined by passing in 'dataset_names' multiple times with
    the correct datasets directory name.

    Visualising the batch and epoch metrics during training requires having a visdom
    server running on port 8998 (and vis=True).

    Args:
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells
        exp_name: name of the experiment directory to save results to
        annot_dir: relative path to annotations
        dataset_names: name of directory containing one datasets
        model_name: architecture name (currently 'resnet-50 or 'inceptionresnetv2')
        pre_trained: path to pretrained weights if starting from local weights
        num_workers: number of workers for parallel processing
        epochs: number of epochs to train for
        batch: batch size of the training set
        val_batch: batch size of the validation sets
        learning_rate: initial learning rate which decreases every step size
        decay_gamma: amount to decay learning rate by. Set to 1 for no decay.
        step_size: epoch at which to apply learning rate decay.
        init_from_inc: whether to use imagenet/coco pretrained weights
        frozen: whether to freeze most of the layers. True for only fine-tuning
        weighted_loss: whether to use weighted loss for imbalanced datasets
        vis: whether to use visdom for visualisation
        get_cuda_device_num: True if you want the code to choose a gpu
    """
    device = get_device(get_cuda_device_num)

    hp = Hyperparameters(
        exp_name,
        annot_dir,
        dataset_names,
        model_name,
        pre_trained,
        epochs,
        batch,
        learning_rate,
        init_from_inc,
        frozen,
    )
    organ = get_organ(organ_name)
    multiple_val_sets = True if len(hp.dataset_names) > 1 else False
    project_dir = get_project_dir(project_name)
    os.chdir(str(project_dir))

    # Setup the model. Can be pretrained from coco or own weights.
    model, image_size = cell_train.setup_model(
        model_name, hp.init_from_inc, len(organ.cells), hp.pre_trained, frozen, device
    )

    # Get all datasets and dataloaders, including separate validation datasets
    dataloaders = cell_train.setup_data(
        organ,
        project_dir / annot_dir,
        hp,
        image_size,
        num_workers,
        multiple_val_sets,
        val_batch,
    )

    # Setup recording of stats for each datasets per batch and epoch
    logger = Logger(list(dataloaders.keys()), ["loss", "accuracy"], vis)

    # Setup training parameters
    optimizer, criterion, scheduler = cell_train.setup_training_params(
        model,
        hp.learning_rate,
        dataloaders["train"],
        device,
        weighted_loss,
        decay_gamma,
        step_size,
    )

    # Save each run by it's timestamp
    run_path = utils.setup_run(project_dir, exp_name, "cell_class")
    hp.to_csv(run_path)

    # train!
    try:
        print(f"Num training images: {len(dataloaders['train'].dataset)}")
        print(
            f"Training on datasets {hp.dataset_names} for {hp.epochs} epochs, "
            f"with lr of {hp.learning_rate}, batch size {hp.batch}, "
            f"init from coco is {hp.init_from_inc}"
        )
        cell_train.train(
            organ,
            hp.epochs,
            model,
            dataloaders,
            optimizer,
            criterion,
            logger,
            scheduler,
            run_path,
            device,
        )
    except KeyboardInterrupt:
        save = input("Would you like to save hyperparameters, train stats, and model anyway? y/n: ")
        if save == "y":
            # Save hyperparameters, the logged train stats, and the final model
            cell_train.save_state(logger, model, hp, run_path)

    # Save hyperparameters, the logged train stats, and the final model
    cell_train.save_state(logger, model, hp, run_path)


if __name__ == "__main__":
    typer.run(main)
