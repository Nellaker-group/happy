from enum import Enum


import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

from happy.train.utils import plot_confusion_matrix, get_tissue_confusion_matrix
from happy.models.clustergcn import ClusterGCN
from happy.graph.create_graph import get_nodes_within_tiles


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"
    intersection = "intersection"


def setup_node_splits(
    data,
    tissue_class,
    mask_unlabelled,
    val_patch_files=[],
    test_patch_files=[],
    verbose=True,
):
    all_xs = data["pos"][:, 0]
    all_ys = data["pos"][:, 1]

    # Mark everything as training data first
    train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.train_mask = train_mask

    # Mask unlabelled data to ignore during training
    if mask_unlabelled and tissue_class is not None:
        unlabelled_inds = (tissue_class == 0).nonzero()[0]
        unlabelled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        unlabelled_mask[unlabelled_inds] = True
        data.unlabelled_mask = unlabelled_mask
        train_mask[unlabelled_inds] = False
        data.train_mask = train_mask
        if verbose:
            print(f"{len(unlabelled_inds)} nodes marked as unlabelled")

    # Split the graph by masks into training, validation and test nodes
    if len(val_patch_files) > 0:
        if verbose:
            print("Splitting graph by validation patch")
        val_node_inds = []
        for file in val_patch_files:
            patches_df = pd.read_csv(file)
            for row in patches_df.itertuples(index=False):
                if row.x == 0 and row.y == 0 and row.width == -1 and row.height == -1:
                    data.val_mask = torch.ones(data.num_nodes, dtype=torch.bool)
                    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                    if mask_unlabelled and tissue_class is not None:
                        data.val_mask[unlabelled_inds] = False
                        data.train_mask[unlabelled_inds] = False
                        data.test_mask[unlabelled_inds] = False
                    if verbose:
                        print(
                            f"All nodes marked as validation: "
                            f"{data.val_mask.sum().item()}"
                        )
                    return data
                val_node_inds.extend(
                    get_nodes_within_tiles(
                        (row.x, row.y), row.width, row.height, all_xs, all_ys
                    )
                )
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask[val_node_inds] = True
        train_mask[val_node_inds] = False
        if len(test_patch_files) > 0:
            test_node_inds = []
            for file in test_patch_files:
                patches_df = pd.read_csv(file)
                for row in patches_df.itertuples(index=False):
                    test_node_inds.extend(
                        get_nodes_within_tiles(
                            (row.x, row.y), row.width, row.height, all_xs, all_ys
                        )
                    )
            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            test_mask[test_node_inds] = True
            train_mask[test_node_inds] = False
            data.test_mask = test_mask
        else:
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = val_mask
        data.train_mask = train_mask
        if verbose:
            print(
                f"Graph split into {data.train_mask.sum().item()} train nodes "
                f"and {data.val_mask.sum().item()} validation nodes "
                f"and {data.test_mask.sum().item()} test nodes"
            )
    else:
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    return data


def setup_model(
    data, device, layers, num_classes, hidden_units, dropout=None, pretrained=None
):
    if pretrained:
        return torch.load(pretrained / "graph_model.pt", map_location=device)
    model = ClusterGCN(
        data.num_node_features,
        hidden_channels=hidden_units,
        out_channels=num_classes,
        num_layers=layers,
        dropout=dropout,
        reduce_dims=64,
    )
    model = model.to(device)
    return model


def setup_training_params(
    model,
    organ,
    learning_rate,
    train_dataloader,
    device,
    weighted_loss,
    use_custom_weights,
):
    if weighted_loss:
        data_classes = train_dataloader.cluster_data.data.y[
            train_dataloader.cluster_data.data.train_mask
        ].numpy()
        class_weights = _compute_tissue_weights(data_classes, organ, use_custom_weights)
        class_weights = torch.FloatTensor(class_weights)
        class_weights = class_weights.to(device)
        criterion = torch.nn.NLLLoss(weight=class_weights)
    else:
        criterion = torch.nn.NLLLoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimiser, criterion


def train(model, optimiser, criterion, train_loader, device):
    model.train()

    total_loss = 0
    total_examples = 0
    total_correct = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        out = model(batch.x, batch.edge_index)
        train_out = out[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        loss = criterion(train_out, train_y)
        loss.backward()
        optimiser.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_correct += int(train_out.argmax(dim=-1).eq(train_y).sum().item())
        total_examples += nodes

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def validate(model, data, eval_loader, device):
    model.eval()
    out, _ = model.inference(data.x, eval_loader, device)
    out = out.argmax(dim=-1)
    y = data.y.to(out.device)
    train_accuracy = int((out[data.train_mask].eq(y[data.train_mask])).sum()) / int(
        data.train_mask.sum()
    )
    val_accuracy = int((out[data.val_mask].eq(y[data.val_mask])).sum()) / int(
        data.val_mask.sum()
    )
    return train_accuracy, val_accuracy


@torch.no_grad()
def inference(model, x, eval_loader, device):
    print("Running inference")
    model.eval()
    out, graph_embeddings = model.inference(x, eval_loader, device)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
    predicted_labels = predicted_labels.cpu().numpy()
    out = out.cpu().detach().numpy()
    return out, graph_embeddings, predicted_labels


def evaluate(tissue_class, predicted_labels, out, organ, remove_unlabelled):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    if remove_unlabelled:
        tissue_ids = tissue_ids[1:]

    accuracy = accuracy_score(tissue_class, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(tissue_class, predicted_labels)
    top_2_accuracy = top_k_accuracy_score(tissue_class, out, k=2, labels=tissue_ids)
    top_3_accuracy = top_k_accuracy_score(tissue_class, out, k=3, labels=tissue_ids)
    f1_macro = f1_score(tissue_class, predicted_labels, average="macro")
    roc_auc = roc_auc_score(
        tissue_class,
        softmax(out, axis=-1),
        average="macro",
        multi_class="ovo",
        labels=tissue_ids,
    )
    weighted_roc_auc = roc_auc_score(
        tissue_class,
        softmax(out, axis=-1),
        average="weighted",
        multi_class="ovo",
        labels=tissue_ids,
    )
    print("-----------------------")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Top 2 accuracy: {top_2_accuracy:.6f}")
    print(f"Top 3 accuracy: {top_3_accuracy:.6f}")
    print(f"Balanced accuracy: {balanced_accuracy:.6f}")
    print(f"F1 macro score: {f1_macro:.6f}")
    print(f"ROC AUC macro: {roc_auc:.6f}")
    print(f"Weighted ROC AUC macro: {weighted_roc_auc:.6f}")
    print("-----------------------")


def evaluation_plots(tissue_class, predicted_labels, organ, run_path):
    # Order by counts and category
    sort_inds = [1, 2, 4, 0, 3, 5, 6, 7, 8]

    tissue_mapping = {tissue.id: tissue.name for tissue in organ.tissues}
    tissue_colours = {tissue.id: tissue.colour for tissue in organ.tissues}

    recalls = recall_score(tissue_class, predicted_labels, average=None)[sort_inds]
    precisions = precision_score(tissue_class, predicted_labels, average=None)[
        sort_inds
    ]
    print("Plotting recall and precision bar plots")
    plt.rcParams["figure.dpi"] = 600
    r_df = pd.DataFrame(recalls)
    plt.figure(figsize=(10, 3))
    sns.set(style="white", font_scale=1.2)
    colours = [tissue_colours[n] for n in np.unique(tissue_class)[sort_inds]]
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
    l_df = pd.DataFrame(tissue_counts[sort_inds])
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
    sorted_labels = [tissue_mapping[n] for n in np.unique(tissue_class)[sort_inds]]
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    plot_confusion_matrix(cm_df, "All Tissues", run_path, "d", reorder=sorted_labels)
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    plot_confusion_matrix(
        cm_df_props, "All Tissues Proportion", run_path, ".2f", reorder=sorted_labels
    )


def collect_params(
    seed,
    organ_name,
    exp_name,
    run_ids,
    x_min,
    y_min,
    width,
    height,
    k,
    graph_method,
    batch_size,
    num_neighbours,
    learning_rate,
    epochs,
    layers,
    weighted_loss,
    custom_weights,
):
    return pd.DataFrame(
        {
            "seed": seed,
            "organ_name": organ_name,
            "exp_name": exp_name,
            "run_ids": [np.array(run_ids)],
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "k": k,
            "graph_method": graph_method,
            "batch_size": batch_size,
            "num_neighbours": num_neighbours,
            "learning_rate": learning_rate,
            "weighted_loss": weighted_loss,
            "custom_weights": custom_weights,
            "epochs": epochs,
            "layers": layers,
        },
        index=[0],
    )


def save_state(run_path, logger, model, epoch):
    torch.save(model, run_path / f"{epoch}_graph_model.pt")
    logger.to_csv(run_path / "graph_train_stats.csv")


def _compute_tissue_weights(data_classes, organ, use_custom_weights):
    unique_classes = np.unique(data_classes)
    if not use_custom_weights:
        weighting = "balanced"
    else:
        custom_weights = [1, 0.85, 0.9, 10.5, 0.8, 1.3, 5.6, 3, 77]
        weighting = dict(zip(list(unique_classes), custom_weights))
    class_weights = compute_class_weight(
        weighting, classes=unique_classes, y=data_classes
    )
    # Account for missing tissues in training data
    classes_in_training = set(unique_classes)
    all_classes = {tissue.id for tissue in organ.tissues}
    missing_classes = list(all_classes - classes_in_training)
    missing_classes.sort()
    for i in missing_classes:
        class_weights = np.insert(class_weights, i, 0.0)
    return class_weights
