from typing import Optional
from pathlib import Path

import typer

from happy.db.models_training import Model, TrainRun
from happy.db.graph_model import GraphModel
import happy.db.eval_runs_interface as db

app = typer.Typer()


@app.command("nucleus-cell")
def add_nucleus_cell_model(
    db_name: str = "main.db",
    path_to_model: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    model_performance: float = typer.Option(...),
    run_name: str = typer.Option(...),
    run_type: str = typer.Option(...),
    path_to_pretrained_model: Optional[str] = None,
    num_epochs: int = typer.Option(...),
    batch_size: int = typer.Option(...),
    init_lr: float = typer.Option(...),
    lr_step: Optional[int] = None,
    model_architecture: str = typer.Option(...),
):
    """Add a trained nucleus or cell model to the database.

    Args:
        path_to_model: absolute path to the saved model weights file
        model_performance: validation performance (0-1)
        run_name: name of the training run
        run_type: one of 'Nuclei' or 'Cell'
        path_to_pretrained_model: path to pretrained weights if used
        num_epochs: number of epochs trained
        batch_size: batch size used during training
        init_lr: initial learning rate
        lr_step: epoch step at which learning rate decayed
        model_architecture: model type used (e.g. 'retinanet' or 'resnet-50')
    """
    db.init(db_name)

    train_run = TrainRun.create(
        run_name=run_name,
        type=run_type,
        pre_trained_path=path_to_pretrained_model or "",
        num_epochs=num_epochs,
        batch_size=batch_size,
        init_lr=init_lr,
        lr_step=lr_step,
    )
    model = Model.create(
        train_run=train_run,
        type=run_type,
        path=path_to_model,
        architecture=model_architecture,
        performance=model_performance,
    )
    print(f"Registered {run_type} model: {path_to_model}")
    print(f"MODEL_ID={model.id}")


@app.command("tissue-model")
def add_tissue_model(
    db_name: str = "main.db",
    path_to_model: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True,
        help="Absolute path to the .pt weights file",
    ),
    exp_name: str = typer.Option(..., help="Experiment name (e.g. 'standardise_emb_bn')"),
    model_type: str = typer.Option(..., help="GNN architecture key (e.g. 'sup_clustergcn')"),
    organ: str = typer.Option(..., help="Organ the model was trained on (e.g. 'placenta')"),
    performance: Optional[float] = typer.Option(None, help="Validation accuracy (0-1)"),
    hyperparameters_path: Optional[Path] = typer.Option(
        None, exists=True, help="Path to train_params.json saved alongside the weights"
    ),
):
    """Add a trained tissue graph model to the database.

    Run after tissue_train.py completes to register the model.

    Example:
        python -m happy.db.add_model tissue-model \\
            --path-to-model /path/to/700_graph_model.pt \\
            --exp-name standardise_emb_bn \\
            --model-type sup_clustergcn \\
            --organ placenta \\
            --performance 0.85
    """
    db.init(db_name)

    graph_model = GraphModel.create(
        path=str(path_to_model),
        hyperparameters_path=str(hyperparameters_path) if hyperparameters_path else None,
        exp_name=exp_name,
        model_type=model_type,
        organ=organ,
        performance=performance,
    )
    print(f"Registered tissue model with ID: {graph_model.id}")
    print(f"  path:        {path_to_model}")
    print(f"  exp_name:    {exp_name}")
    print(f"  model_type:  {model_type}")
    print(f"  organ:       {organ}")
    print(f"  performance: {performance}")


if __name__ == "__main__":
    app()
