from pathlib import Path
from collections import namedtuple
from typing import List
import os

import typer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    recall_score,
    precision_score,
)

from happy.utils.utils import get_device
from happy.train.utils import (
    get_cell_confusion_matrix,
    plot_confusion_matrix,
    plot_cell_pr_curves,
)
from happy.organs import get_organ
from happy.train.cell_train import setup_model
from happy.data.setup_data import setup_cell_datasets
from happy.data.setup_dataloader import setup_dataloaders


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    print_mean_confidence: bool = False,
    plot_pr: bool = True,
    plot_cm: bool = True,
    use_test_set: bool = True,
):
    """Evaluates model performance across validation datasets

    Args:
        project_name: name of the project dir to save visualisations to
        organ_name: name of organ for getting the cells
        annot_dir: relative path to annotations
        pre_trained: relative path to pretrained model
        dataset_names: the datasets who's validation set to evaluate over
        print_mean_confidence: whether to print confidence confusion matrix
        plot_pr: whether to plot the pr curves
        plot_cm: whether to plot the confusion matrix
        use_test_set: whether to use the test set for validation
    
    NB test set should always named as: 'projects/{organ}/datasets/cell_class/{dataset_name}_test'
    ALSO, cell annotation: test_cell.csv should contain the same corresponding file path to your test set
    """
    device = get_device()

    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )
    os.chdir(str(project_dir))

    plot_dir='../../analysis/evaluation/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(dataset_names, 100)
    organ = get_organ(organ_name)
    cell_mapping = {cell.id: cell.label for cell in organ.cells}

    model, image_size = setup_model(
        "resnet-50", False, len(organ.cells), pre_trained, False, device
    )
    model.eval()

    multiple_val_sets = True if len(dataset_names) > 1 else False
    datasets = setup_cell_datasets(
        organ,
        project_dir / annot_dir,
        hp.dataset_names,
        image_size,
        multiple_val_sets,
        use_test_set,
    )
    if not use_test_set:
        datasets.pop("train")
    dataloaders = setup_dataloaders(False, datasets, 3, hp.batch, hp.batch)


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
        balanced_accuracy = balanced_accuracy_score(
            ground_truth[dataset_name], predictions[dataset_name]
        )
        cohen_kappa = cohen_kappa_score(
            ground_truth[dataset_name], predictions[dataset_name]
        )
        mcc = matthews_corrcoef(ground_truth[dataset_name], predictions[dataset_name])
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
        print(f"Balanced Accuracy: {balanced_accuracy:.6f}")
        print(f"Top 2 accuracy: {top_2_accuracy:.6f}")
        print(f"Cohen's Kappa: {cohen_kappa:.6f}")
        print(f"MCC: {mcc:.6f}")
        print(f"ROC AUC macro: {roc_auc:.6f}")

        alt_ground_truth = _convert_to_alt_label(organ, ground_truth[dataset_name])
        alt_predictions = _convert_to_alt_label(organ, predictions[dataset_name])
        alt_label_accuracy = accuracy_score(alt_ground_truth, alt_predictions)
        print(f"Alt Cell Label Accuracy: {alt_label_accuracy:.3f}")

        if print_mean_confidence:
            print("Mean confidence across cells for ground truth cell types:")
            print(list(cell_mapping.values()))
            for cell_id, cell_label in cell_mapping.items():
                cell_inds = (np.array(ground_truth[dataset_name]) == cell_id).nonzero()[
                    0
                ]
                cell_scores = np.array(outs[dataset_name])[cell_inds]
                cell_confidences = np.mean(softmax(cell_scores, axis=1), axis=0)
                if np.all(np.isnan(cell_confidences)):
                    continue
                print(f"{cell_label}: {np.round(cell_confidences, 3)}")

        if plot_pr:
            save_path = f"../../analysis/evaluation/plots/{dataset_name}_pr_curves.png"
            plot_cell_pr_curves(
                organ,
                ground_truth[dataset_name],
                outs[dataset_name],
                save_path,
            )

        if plot_cm:
            cell_mapping = {cell.id: cell.name for cell in organ.cells}
            cell_colours = {cell.id: cell.colour for cell in organ.cells}

            sorted_labels = [
                cell_mapping[n]
                for n in np.array(list(cell_mapping.keys()))
            ]

            _plot_confusion_matrix(
                organ,
                predictions[dataset_name],
                ground_truth[dataset_name],
                dataset_name,
                reorder=sorted_labels,
            )

            if plot_pr:
                recalls = recall_score(
                    ground_truth[dataset_name], predictions[dataset_name], average=None
                )
                precisions = precision_score(
                    ground_truth[dataset_name], predictions[dataset_name], average=None
                )
                print("Plotting recall and precision bar plots")
                plt.rcParams["figure.dpi"] = 600
                r_df = pd.DataFrame(recalls)
                plt.figure(figsize=(10, 3))
                sns.set(style="white", font_scale=1.2)
                colours = [
                    cell_colours[n]
                    for n in np.unique(ground_truth[dataset_name])
                ]
                ax = sns.barplot(data=r_df.T, palette=colours)
                ax.set(ylabel="Recall", xticklabels=[], ylim=[0.0, 1.0])
                ax.tick_params(bottom=False)
                sns.despine(bottom=True)
                plt.savefig("../../analysis/evaluation/plots/recalls.png")
                plt.close()
                plt.clf()

                p_df = pd.DataFrame(precisions)
                plt.rcParams["figure.dpi"] = 600
                plt.figure(figsize=(3, 10))
                sns.set(style="white", font_scale=1.2)
                ax = sns.barplot(data=p_df.T, palette=colours, orient="h")
                ax.set(xlabel="Precision", yticklabels=[], xlim=[0.0, 1.0])
                ax.tick_params(left=False)
                sns.despine(left=True)
                plt.savefig("../../analysis/evaluation/plots/precisions.png")
                plt.close()
                plt.clf()

                print("Plotting cell counts bar plot")
                _, cell_counts = np.unique(ground_truth[dataset_name], return_counts=True)
                l_df = pd.DataFrame(cell_counts)
                plt.rcParams["figure.dpi"] = 600
                plt.figure(figsize=(10, 3))
                sns.set(style="white", font_scale=1.2)
                ax = sns.barplot(data=l_df.T, palette=colours)
                ax.set(ylabel="Count", xticklabels=[])
                ax.tick_params(bottom=False)
                sns.despine(bottom=True)
                plt.savefig("../../analysis/evaluation/plots/cell_counts.png")
                plt.close()
                plt.clf()


def _convert_to_alt_label(organ, labels):
    ids = [cell.id for cell in organ.cells]
    alt_ids = [cell.alt_id for cell in organ.cells]
    alt_id_mapping = dict(zip(ids, alt_ids))
    return [alt_id_mapping[label] for label in labels]


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
