from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import typer
from torch_geometric.transforms import RandomNodeSplit

from happy.graph.graph_creation.create_graph import get_nodes_within_tiles


def main(
    data: List[str] = typer.Option(..., help="Tissue annotation CSV files (px,py,class), one per slide"),
    include_validation: bool = typer.Option(True, help="Whether to create a validation split"),
    num_val: float = typer.Option(0.15, help="Proportion of annotation points for validation"),
    num_test: float = typer.Option(0.15, help="Proportion of annotation points for test"),
    output_dir: str = typer.Option("project/placenta/results/tissue_annots/splits", help="Directory to save split annotation files"),
):
    """Split tissue annotation CSV files into train/val/test subsets.

    Loads each annotation CSV, randomly assigns points to train/val/test splits
    by the given proportions, and saves separate output CSVs per split.
    Output files are named <stem>_train.csv, <stem>_val.csv, <stem>_test.csv.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for csv_path in data:
        path = Path(csv_path)
        df = pd.read_csv(path, sep=None, engine="python")
        n = len(df)

        indices = np.random.permutation(n)
        n_test = int(n * num_test) if include_validation else 0
        n_val = int(n * num_val) if include_validation else 0

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        df.iloc[train_idx].to_csv(out_path / f"{path.stem}_train.csv", index=False)
        print(f"{path.name}: {len(train_idx)} train points -> {path.stem}_train.csv")

        if include_validation:
            df.iloc[val_idx].to_csv(out_path / f"{path.stem}_val.csv", index=False)
            df.iloc[test_idx].to_csv(out_path / f"{path.stem}_test.csv", index=False)
            print(f"{path.name}: {len(val_idx)} val points -> {path.stem}_val.csv")
            print(f"{path.name}: {len(test_idx)} test points -> {path.stem}_test.csv")


def setup_splits_by_runid(
    data, tissue_class, include_validation, val_patch_files, test_patch_files,
    num_val=0.15, num_test=0.15,
):
    """Split graph nodes into train/val/test masks.

    If val_patch_files or test_patch_files are provided, those CSV files define
    the spatial regions for each split. Otherwise, nodes are split randomly
    (if include_validation=True) or the whole graph is used for training.
    """
    print("setting up graph nodes for model training")
    return setup_node_splits(
        data,
        tissue_class,
        True,
        include_validation,
        val_patch_files,
        test_patch_files,
        num_val=num_val,
        num_test=num_test,
    )


def setup_node_splits(
    data,
    tissue_class,
    mask_unlabelled,
    include_validation=True,
    val_patch_files=[],
    test_patch_files=[],
    num_val=0.15,
    num_test=0.15,
    verbose=True,
):
    all_xs = data["pos"][:, 0]
    all_ys = data["pos"][:, 1]

    # Mark everything as training data first
    train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.train_mask = train_mask

    # Mask unlabelled data to ignore during training
    if mask_unlabelled and tissue_class is not None:
        #tissue class is just used to mask unlabelled nodes (class 0)
        unlabelled_inds = (tissue_class == 0).nonzero()[0]
        unlabelled_inds = unlabelled_inds[unlabelled_inds < data.num_nodes]
        unlabelled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        unlabelled_mask[unlabelled_inds] = True
        data.unlabelled_mask = unlabelled_mask
        train_mask[unlabelled_inds] = False
        data.train_mask = train_mask
        if verbose:
            print(f"{len(unlabelled_inds)} nodes marked as unlabelled")

    # Split the graph by masks into training, validation and test nodes
    if include_validation:
        if len(val_patch_files) == 0:
            if len(test_patch_files) == 0:
                if verbose:
                    print("No validation patch provided, splitting nodes randomly")
                data = RandomNodeSplit(num_val=num_val, num_test=num_test)(data)
            else:
                if verbose:
                    print(
                        "No validation patch provided, splitting nodes randomly into "
                        "train and val and using test patch"
                    )
                data = RandomNodeSplit(num_val=num_val, num_test=0)(data)
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
                data.val_mask[test_node_inds] = False
                data.train_mask[test_node_inds] = False
                data.test_mask = test_mask
            if mask_unlabelled and tissue_class is not None:
                data.val_mask[unlabelled_inds] = False
                data.train_mask[unlabelled_inds] = False
                data.test_mask[unlabelled_inds] = False
        else:
            if verbose:
                print("Splitting graph by validation patch")
            val_node_inds = []
            for file in val_patch_files:
                patches_df = pd.read_csv(file)
                for row in patches_df.itertuples(index=False):
                    if (
                        row.x == 0
                        and row.y == 0
                        and row.width == -1
                        and row.height == -1
                    ):
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
            if mask_unlabelled and tissue_class is not None:
                data.val_mask[unlabelled_inds] = False
                data.train_mask[unlabelled_inds] = False
                data.test_mask[unlabelled_inds] = False
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


if __name__ == "__main__":
    typer.run(main)