import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    recall_score,
    precision_score,
)

from happy.train.utils import (
    get_tissue_confusion_matrix,
    plot_confusion_matrix,
    plot_tissue_pr_curves,
)


def evaluate(tissue_class, predicted_labels, out, organ, compress_labels=False):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    # remove unlabelled tissues
    tissue_ids = tissue_ids[1:]

    if compress_labels:
        tissue_class, predicted_labels, out = _compress_tissue_labels(
            organ, tissue_class, predicted_labels, out
        )

    accuracy = accuracy_score(tissue_class, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(tissue_class, predicted_labels)
    f1_macro = f1_score(tissue_class, predicted_labels, average="macro")
    cohen_kappa = cohen_kappa_score(tissue_class, predicted_labels)
    mcc = matthews_corrcoef(tissue_class, predicted_labels)
    if not compress_labels:
        top_2_accuracy = top_k_accuracy_score(tissue_class, out, k=2, labels=tissue_ids)
        top_3_accuracy = top_k_accuracy_score(tissue_class, out, k=3, labels=tissue_ids)
        roc_auc = roc_auc_score(
            tissue_class,
            softmax(out, axis=-1),
            average="macro",
            multi_class="ovo",
            labels=tissue_ids,
        )
        if len(np.unique(tissue_class)) > 1:
            weighted_roc_auc = roc_auc_score(
                tissue_class,
                softmax(out, axis=-1),
                average="weighted",
                multi_class="ovo",
                labels=tissue_ids,
            )
        else:
            print("Only one class present in ground truth, skipping weighted ROC AUC")
            weighted_roc_auc = 0

    print("-----------------------")
    print(f"Accuracy: {accuracy:.6f}")
    if not compress_labels:
        print(f"Top 2 accuracy: {top_2_accuracy:.6f}")
        print(f"Top 3 accuracy: {top_3_accuracy:.6f}")
        print(f"ROC AUC macro: {roc_auc:.6f}")
        print(f"Weighted ROC AUC macro: {weighted_roc_auc:.6f}")
    print(f"Balanced accuracy: {balanced_accuracy:.6f}")
    print(f"F1 macro score: {f1_macro:.6f}")
    print(f"Cohen's Kappa score: {cohen_kappa:.6f}")
    print(f"MCC score: {mcc:.6f}")
    print("-----------------------")


def evaluate_basic(tissue_class, predicted_labels, organ, remove_first=False):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    # remove unlabelled tissues
    if remove_first:
        tissue_ids = tissue_ids[1:]

    accuracy = accuracy_score(tissue_class, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(tissue_class, predicted_labels)

    print("-----------------------")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Balanced accuracy: {balanced_accuracy:.6f}")
    print("-----------------------")


def evaluation_plots(tissue_class, predicted_labels, out, organ, run_path):
    # Order by counts and category

    tissue_mapping = {tissue.id: tissue.name for tissue in organ.tissues}
    tissue_colours = {tissue.id: tissue.colour for tissue in organ.tissues}

    recalls = recall_score(tissue_class, predicted_labels, average=None)
    precisions = precision_score(tissue_class, predicted_labels, average=None)
    print("Plotting recall and precision bar plots")
    plt.rcParams["figure.dpi"] = 600
    r_df = pd.DataFrame(recalls)
    plt.figure(figsize=(10, 3))
    sns.set(style="white", font_scale=1.2)
    colours = [tissue_colours[n] for n in np.unique(tissue_class)]
    ax = sns.barplot(data=r_df.T, palette=colours)
    ax.set(ylabel="Recall", xticklabels=[])
    ax.tick_params(bottom=False)
    sns.despine(bottom=True)
    plt.savefig(run_path / "recalls.png")
    plt.close()
    plt.clf()

    p_df = pd.DataFrame(precisions)
    plt.rcParams["figure.dpi"] = 600
    plt.figure(figsize=(3, 10))
    sns.set(style="white", font_scale=1.2)
    ax = sns.barplot(data=p_df.T, palette=colours, orient="h")
    ax.set(xlabel="Precision", yticklabels=[])
    ax.tick_params(left=False)
    sns.despine(left=True)
    plt.savefig(run_path / "precisions.png")
    plt.close()
    plt.clf()

    print("Plotting tissue counts bar plot")
    _, tissue_counts = np.unique(tissue_class, return_counts=True)
    l_df = pd.DataFrame(tissue_counts)
    plt.rcParams["figure.dpi"] = 600
    plt.figure(figsize=(10, 3))
    sns.set(style="white", font_scale=1.2)
    ax = sns.barplot(data=l_df.T, palette=colours)
    ax.set(ylabel="Count", xticklabels=[])
    ax.tick_params(bottom=False)
    sns.despine(bottom=True)
    plt.savefig(run_path / "tissue_counts.png")
    plt.close()
    plt.clf()

    print("Plotting confusion matrices")
    cm_df, cm_df_props = get_tissue_confusion_matrix(
        organ, predicted_labels, tissue_class, proportion_label=False
    )
    sorted_labels = [tissue_mapping[n] for n in np.unique(tissue_class)]
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    plot_confusion_matrix(cm_df, "All Tissues", run_path, "d", reorder=sorted_labels)
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    plot_confusion_matrix(
        cm_df_props, "All Tissues Proportion", run_path, ".2f", reorder=sorted_labels
    )

    print("Plotting pr curves")
    plot_tissue_pr_curves(
        tissue_mapping,
        tissue_colours,
        tissue_class,
        predicted_labels,
        out,
        run_path / "pr_curves.png",
    )


def _compress_tissue_labels(organ, tissue_class, predicted_labels, out):
    compression_mapping = {tissue.id: tissue.alt_id for tissue in organ.tissues}

    tissue_class = np.array([compression_mapping[t] for t in tissue_class])
    predicted_labels = np.array([compression_mapping[t] for t in predicted_labels])
    # todo: add compression for out

    return tissue_class, predicted_labels, out
