import time
from pathlib import Path
from typing import List, Optional

import typer

from happy.train.nuc_train import YoloConfig, train_yolo, merge_yamls, read_best_metrics
from happy.train.hyperparameters import NucHyperparameters
from happy.train import utils
from happy.utils.utils import get_project_dir
from happy.db.models_training import Model, TrainRun
import happy.db.eval_runs_interface as db


def main(
    project_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    data: List[str] = typer.Option(..., help="Path to dataset YAML. Pass multiple times to train a joint model across organs."),
    model_name: str = typer.Option("yolo26n.pt", help="YOLO model variant or path to weights"),
    pre_trained: Optional[str] = typer.Option(None, help="Path to local weights to resume from (instead model name)"),
    num_workers: int = typer.Option(4),
    epochs: int = typer.Option(100),
    batch: int = typer.Option(8),
    imgsz: int = typer.Option(1280, help="Image size (must be multiple of 32)"),
    learning_rate: float = typer.Option(0.001),
    weight_decay: float = typer.Option(0.0005),
    patience: int = typer.Option(20, help="Early stopping patience in epochs"),
    optimiser: str = typer.Option("AdamW", help="Optimiser: AdamW or SGD"),
    device: str = typer.Option("cuda"),
    single_cls: bool = typer.Option(True, help="Treat all classes as one (e.g. as with single nuc detection)"),
    seed: int = typer.Option(0, help="Random seed for reproducibility"),
    add_to_db: bool = typer.Option(False, help="Save nuc model to database"),
    get_cuda_device_num: bool = typer.Option(False, help="Auto-select GPU device"),
):
    """Train a nuclei detection model.

    To resume, give pre trained model path instead of model name

    Args:
        project_name: name of the project dir to save results to
        exp_name: name of the experiment directory to save results to
        data: path to the dataset YAML file defining train/val/test splits from the project dir.
            give multiple flags with new ds for multidataset training
        model_name: YOLO model variant (e.g. yolo26n.pt, yolo26s.pt) or path to weights
        pre_trained: path to local weights to resume from instead of model_name
        num_workers: number of dataloader workers
        epochs: number of epochs to train for
        batch: batch size
        imgsz: input image size (must be a multiple of 32) - maxs square and black pads, bigger = more memory onto gpu
        learning_rate: initial learning rate
        weight_decay: L2 regularisation
        patience: early stopping patience
        optimiser: AdamW or SGD
        device: cuda, cpu, or GPU index
        add_to_db: save nuc model to database
        get_cuda_device_num: auto-select a free GPU or give number
    """

    # Resolve paths
    data = [str(Path(d).resolve()) for d in data]
    if pre_trained:
        pre_trained = str(Path(pre_trained).resolve())

    if get_cuda_device_num:
        import torch
        device = str(torch.cuda.current_device())

    if add_to_db:
        db.init("main.db")

    hp = NucHyperparameters(
        exp_name=exp_name,
        data="|".join(data),
        model_name=model_name,
        pre_trained=pre_trained,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        optimiser=optimiser,
        workers=num_workers,
        seed=seed,
    )

    project_dir = get_project_dir(project_name)
    run_path = utils.setup_run(project_dir, exp_name, "nuclei")
    hp.to_csv(run_path)

    # merge yamls if multiple datasets
    data_yaml = merge_yamls(data, run_path) if len(data) > 1 else data[0]

    cfg = YoloConfig(
        data=data_yaml,
        model_name=hp.model_name,
        pre_trained=hp.pre_trained,
        epochs=hp.epochs,
        imgsz=hp.imgsz,
        batch=hp.batch,
        device=device,
        workers=hp.workers,
        optimizer=hp.optimiser,
        lr0=hp.learning_rate,
        weight_decay=hp.weight_decay,
        patience=hp.patience,
        project=project_name,
        name=exp_name,
        single_cls=single_cls,
        seed=hp.seed,
    )

    print(
        f"Training YOLO nuclei detection for {hp.epochs} epochs, "
        f"lr={hp.learning_rate}, batch={hp.batch}, imgsz={hp.imgsz}, "
        f"model={hp.model_name}"
    )

    start = time.time()
    try:
        best_weights, last_weights = train_yolo(cfg, run_path)
    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            hp.to_csv(run_path)
        return
    train_time = time.time() - start
    print(f"Training time: {train_time:.1f}s ({train_time / 60:.1f} min)")

    if add_to_db:
        best_map50, num_epochs = read_best_metrics(run_path)
        train_run = TrainRun.create(
            run_name=exp_name,
            type="Nuclei",
            pre_trained_path=pre_trained or "",
            num_epochs=num_epochs,
            batch_size=hp.batch,
            init_lr=hp.learning_rate,
            lr_step=None,
        )
        model_record = Model.create(
            train_run=train_run,
            type="Nuclei",
            path=str(best_weights.resolve()),
            architecture="yolo26",
            performance=round(best_map50, 4),
        )
        print(f"Model registered in DB with ID: {model_record.id}")

    print(f"Best weights: {best_weights}")
    print(f"Last weights: {last_weights}")
    print(f"Run saved to: {run_path}")


if __name__ == "__main__":
    typer.run(main)
