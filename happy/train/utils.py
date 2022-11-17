from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def setup_run(project_dir, exp_name, dataset_type):
    fmt = "%Y-%m-%dT%H-%M-%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = project_dir / "results" / dataset_type / exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def get_cell_confusion_matrix(organ, pred, truth, proportion_label=False):
    cell_labels = [cell.name for cell in organ.cells]
    cell_ids = {cell.id for cell in organ.cells}

    unique_values_in_pred = set(pred)
    unique_values_in_truth = set(truth)
    unique_values_in_matrix = unique_values_in_pred.union(unique_values_in_truth)
    missing_cell_ids = list(cell_ids - unique_values_in_matrix)
    missing_cell_ids.sort()

    cm = confusion_matrix(truth, pred)

    if len(missing_cell_ids) > 0:
        for missing_id in missing_cell_ids:
            column_insert = np.zeros((cm.shape[0], 1))
            cm = np.hstack((cm[:, :missing_id], column_insert, cm[:, missing_id:]))
            row_insert = np.zeros((1, cm.shape[1]))
            cm = np.insert(cm, missing_id, row_insert, 0)

    if proportion_label:
        row_labels = []
        unique_counts = cm.sum(axis=1)
        total_counts = cm.sum()
        label_proportions = ((unique_counts / total_counts) * 100).round(2)
        for i, label in enumerate(cell_labels):
            row_labels.append(f"{label}\n({label_proportions[i]}%)")
    else:
        row_labels = cell_labels

    cm_df = pd.DataFrame(cm, columns=cell_labels, index=row_labels).astype(int)
    args_to_sort = np.argsort([cell.structural_id for cell in organ.cells])
    cm_df = cm_df[cm_df.columns[args_to_sort]]
    cm_df = cm_df.reindex(cm_df.index[args_to_sort])

    return cm_df


def get_tissue_confusion_matrix(organ, pred, truth, proportion_label=False):
    tissue_ids = {tissue.id for tissue in organ.tissues}
    tissue_labels = [tissue.name for tissue in organ.tissues]

    unique_values_in_pred = set(pred)
    unique_values_in_truth = set(truth)
    unique_values_in_matrix = unique_values_in_pred.union(unique_values_in_truth)
    missing_tissue_ids = list(tissue_ids - unique_values_in_matrix)
    missing_tissue_ids.sort()

    cm = confusion_matrix(truth, pred)

    if len(missing_tissue_ids) > 0:
        for missing_id in missing_tissue_ids:
            column_insert = np.zeros((cm.shape[0], 1))
            cm = np.hstack((cm[:, :missing_id], column_insert, cm[:, missing_id:]))
            row_insert = np.zeros((1, cm.shape[1]))
            cm = np.insert(cm, missing_id, row_insert, 0)

    row_labels = []
    if proportion_label:
        unique_counts = cm.sum(axis=1)
        total_counts = cm.sum()
        label_proportions = ((unique_counts / total_counts) * 100).round(2)
        for i, label in enumerate(tissue_labels):
            row_labels.append(f"{label} ({label_proportions[i]}%)")

    cm_df = pd.DataFrame(cm, columns=tissue_labels, index=tissue_labels).astype(int)
    unique_counts = cm.sum(axis=1)

    cm_df_props = (
        pd.DataFrame(
            cm / unique_counts[:, None], columns=tissue_labels, index=tissue_labels
        )
        .fillna(0)
        .astype(float)
    )

    non_empty_rows = (cm_df.T != 0).any()
    cm_df = cm_df[non_empty_rows]
    cm_df_props = cm_df_props[non_empty_rows]
    empty_row_names = non_empty_rows[non_empty_rows == False].index.tolist()
    cm_df = cm_df.drop(columns=empty_row_names)
    cm_df_props = cm_df_props.drop(columns=empty_row_names)

    if proportion_label:
        row_labels = np.array(row_labels)
        row_labels = row_labels[non_empty_rows]
        cm_df_props.set_index(row_labels, drop=True, inplace=True)

    return cm_df, cm_df_props


def plot_confusion_matrix(cm, dataset_name, run_path, fmt="d", reorder=None):
    save_path = run_path / f"{dataset_name}_confusion_matrix.png"
    if reorder is not None:
        cm = cm[reorder]
        cm = cm.reindex(reorder)
    plt.rcParams["figure.dpi"] = 600
    sns.heatmap(cm, annot=True, cmap="Blues", square=True, cbar=False, fmt=fmt)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.clf()
