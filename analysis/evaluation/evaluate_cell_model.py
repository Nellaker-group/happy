from pathlib import Path
from collections import namedtuple
from typing import List
import os

import typer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score

from happy.utils.utils import get_device, get_project_dir
from happy.train.utils import get_cell_confusion_matrix, plot_confusion_matrix
from happy.organs import get_organ
from happy.train.cell_train import setup_data, setup_model


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    plot_cm: bool = True,
    use_test_set: bool = False,
):
    """Evaluates model performance across validation or test datasets

    Args:
        project_name: name of the project dir to save visualisations to
        organ_name: name of organ for getting the cells
        annot_dir: relative path to annotations
        pre_trained: relative path to pretrained model
        dataset_names: the datasets' validation set to evaluate over
        plot_cm: whether to plot the confusion matrix
        use_test_set: whether to use the test set for validation
    """
    device = get_device()

    project_dir = get_project_dir(project_name)
    os.chdir(str(project_dir))

    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(dataset_names, 100)
    organ = get_organ(organ_name)
    cell_mapping = {cell.id: cell.label for cell in organ.cells}

    model = setup_model(False, len(organ.cells), pre_trained, False, device)
    model.eval()

    multiple_val_sets = True if len(dataset_names) > 1 else False
    dataloaders = setup_data(
        organ,
        project_dir / annot_dir,
        hp,
        (224, 224),
        3,
        multiple_val_sets,
        hp.batch,
        use_test_set,
    )
    if not use_test_set:
        dataloaders.pop("train")

    print("Running inference across datasets")
    predictions = {}
    ground_truth = {}
    outs = {}
    with torch.no_grad():
        for dataset_name in dataloaders:
            predictions[dataset_name] = []
            ground_truth[dataset_name] = []
            outs[dataset_name] = []

            for data in tqdm(dataloaders[dataset_name]):
                ground_truths = data["annot"].tolist()

                out = model(data["img"].to(device).float())
                prediction = torch.max(out, 1)[1].cpu().tolist()
                out = out.cpu().detach().numpy()

                ground_truth[dataset_name].extend(ground_truths)
                predictions[dataset_name].extend(prediction)
                outs[dataset_name].extend(out)

    print("Evaluating datasets")
    for dataset_name in dataloaders:
        print(f"{dataset_name}:")
        accuracy = accuracy_score(ground_truth[dataset_name], predictions[dataset_name])
        top_2_accuracy = top_k_accuracy_score(
            ground_truth[dataset_name],
            outs[dataset_name],
            k=2,
            labels=list(cell_mapping.keys()),
        )
        roc_auc = roc_auc_score(
            ground_truth[dataset_name],
            softmax(outs[dataset_name], axis=-1),
            average="macro",
            multi_class="ovo",
            labels=list(cell_mapping.keys()),
        )
        print(f"Accuracy: {accuracy:.6f}")
        print(f"Top 2 accuracy: {top_2_accuracy:.6f}")
        print(f"ROC AUC macro: {roc_auc:.6f}")

        if plot_cm:
            cell_mapping = {cell.id: cell.name for cell in organ.cells}
            _plot_confusion_matrix(
                organ,
                predictions[dataset_name],
                ground_truth[dataset_name],
                dataset_name,
            )


def _plot_confusion_matrix(organ, pred, truth, dataset_name, reorder=None):
    cm = get_cell_confusion_matrix(organ, pred, truth, proportion_label=False)
    plt.clf()
    plt.rcParams["figure.dpi"] = 600
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    plot_confusion_matrix(
        cm,
        dataset_name,
        Path(f"../../analysis/evaluation/plots/"),
        fmt="d",
        reorder=reorder,
    )


if __name__ == "__main__":
    typer.run(main)
