from collections import namedtuple
from typing import List
import os

import typer

from happy.utils.utils import get_device, get_project_dir
from happy.train.point_eval import evaluate_points_over_dataset
from happy.train.nuc_train import setup_data, setup_model


def main(
    project_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    score_threshold: float = 0.4,
    max_detections: int = 500,
    valid_distance: int = 30,
    use_test_set: bool = False,
):
    """Evaluates model performance across validation or test datasets

    Args:
        project_name: name of the project dir to save visualisations to
        annot_dir: relative path to annotations
        pre_trained: relative path to pretrained model
        dataset_names: the datasets' validation set to evaluate over
        score_threshold: the confidence threshold below which to discard predictions
        max_detections: number of maximum detections to save, ordered by score
        valid_distance: distance to gt in pixels for which a prediction is valid
        use_test_set: whether to use the test set for validation
    """
    device = get_device()

    project_dir = get_project_dir(project_name)
    os.chdir(str(project_dir))

    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(dataset_names, 3)

    multiple_val_sets = True if len(dataset_names) > 1 else False
    dataloaders = setup_data(
        project_dir / annot_dir, hp, multiple_val_sets, 3, 1, use_test_set
    )
    if not use_test_set:
        dataloaders.pop("train")
    model = setup_model(False, device, False, pre_trained)
    model.eval()

    mean_precision = {}
    mean_recall = {}
    mean_f1 = {}
    num_empty_predictions = {}
    for dataset_name in dataloaders:
        precision, recall, f1, num_empty = evaluate_points_over_dataset(
            dataloaders[dataset_name],
            model,
            device,
            score_threshold,
            max_detections,
            valid_distance,
        )
        mean_precision[dataset_name] = precision
        mean_recall[dataset_name] = recall
        mean_f1[dataset_name] = f1
        num_empty_predictions[dataset_name] = num_empty

    print(f"Precision: {mean_precision}")
    print(f"Recall: {mean_recall}")
    print(f"F1: {mean_f1}")

    try:
        print(
            f"Number of Predictions in empty images: {num_empty_predictions['empty']}"
        )
    except KeyError:
        pass


if __name__ == "__main__":
    typer.run(main)
