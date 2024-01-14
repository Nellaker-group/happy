from typing import List, Optional
import os

import typer

from happy.train.hyperparameters import Hyperparameters
from happy.utils.utils import get_device, get_project_dir
from happy.logger.logger import Logger
from happy.train import nuc_train, utils


def main(
    project_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    model_name: str = "retinanet",
    pre_trained: Optional[str] = None,
    num_workers: int = 20,
    epochs: int = 20,
    batch: int = 16,
    val_batch: int = 16,
    learning_rate: float = 1e-4,
    decay_gamma: float = 1,
    step_size: int = 20,
    init_from_inc: bool = False,
    frozen: bool = True,
    vis: bool = False,
):
    """For training a nuclei detection model

    Multiple datasets can be combined by passing in 'dataset_names' multiple times with
    the correct datasets directory name.

    Visualising the batch and epoch level training stats requires having a visdom
    server running on port 8998.

    Args:
        project_name: name of the project dir to save results to
        exp_name: name of the experiment directory to save results to
        annot_dir: relative path to annotations
        dataset_names: name of directory containing one datasets
        model_name: architecture name (currently just 'retinanet')
        pre_trained: path to pretrained weights if starting from local weights
        num_workers: number of workers for parallel processing
        epochs: number of epochs to train for
        batch: batch size of the training set
        val_batch: batch size of the validation sets
        learning_rate: learning rate which decreases every 8 epochs
        decay_gamma: amount to decay learning rate by. Set to 1 for no decay.
        step_size: epoch at which to apply learning rate decay.
        init_from_inc: whether to use imagenet pretrained weights
        frozen: whether to freeze most of the layers. True for only fine-tuning
        vis: whether to send stats to visdom for visualisation
    """
    device = get_device()

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
    multiple_val_sets = True if len(hp.dataset_names) > 1 else False
    project_dir = get_project_dir(project_name)
    os.chdir(str(project_dir))

    # Setup the model. Can be pretrained from coco or own weights.
    model = nuc_train.setup_model(hp.init_from_inc, device, frozen, pre_trained)

    # Get all datasets and dataloaders, including separate validation datasets
    dataloaders = nuc_train.setup_data(
        project_dir / annot_dir, hp, multiple_val_sets, num_workers, val_batch
    )

    # Setup training parameters
    optimizer, scheduler = nuc_train.setup_training_params(
        model, hp.learning_rate, decay_gamma, step_size
    )

    # Setup recording of stats per batch and epoch
    logger = Logger(
        list(dataloaders.keys()), ["loss", "Precision", "Recall", "F1"], vis
    )

    # Save each run by it's timestamp
    run_path = utils.setup_run(project_dir, exp_name, "nuclei")

    # train!
    try:
        print(f"Num training images: {len(dataloaders['train'].dataset)}")
        print(
            f"Training on datasets {hp.dataset_names} for {hp.epochs} epochs, "
            f"with lr of {hp.learning_rate}, batch size {hp.batch}, and init from coco "
            f"is {hp.init_from_inc}"
        )
        nuc_train.train(
            epochs, model, dataloaders, optimizer, logger, scheduler, run_path, device
        )
    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            hp.to_csv(run_path)

    # Save hyperparameters, the logged train stats, and the final model
    nuc_train.save_state(logger, model, hp, run_path)


if __name__ == "__main__":
    typer.run(main)
