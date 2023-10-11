from typing import List

import typer
import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.utils.hdf5 import get_datasets_in_patch, get_embeddings_file
from happy.graph.process_knts import process_knt_cells


def main(
    run_ids: List[int] = typer.Option([]),
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: str = "sup_clustergcn",
    group_knts: bool = False,
    trained_with_grouped_knts: bool = False,
    include_counts: bool = False,
):
    """Plot the distribution of cell and tissue types across multiple WSIs.
    This will plot a box and whisker a swarm plot where each point is each WSI.
    """

    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.name for cell in organ.cells}
    project_dir = get_project_dir(project_name)

    cell_prop_dfs = []
    cell_counts_df = []
    tissue_prop_dfs = []
    tissue_counts_df = []
    for run_id in run_ids:
        # Get path to embeddings hdf5 files
        embeddings_path = get_embeddings_file(project_dir, run_id)
        # Get hdf5 datasets contained in specified box/patch of WSI
        predictions, embeddings, cell_coords, confidence = get_datasets_in_patch(
            embeddings_path, 0, 0, -1, -1
        )

        if group_knts:
            (
                predictions,
                embeddings,
                cell_coords,
                confidence,
                inds_to_remove,
            ) = process_knt_cells(
                predictions, embeddings, cell_coords, confidence, organ, 50, 3
            )

        # print cell predictions from hdf5 file
        unique_cells, cell_counts = np.unique(predictions, return_counts=True)
        unique_cell_labels = []
        for label in unique_cells:
            unique_cell_labels.append(cell_label_mapping[label])
        cell_proportions = [
            round((count / sum(cell_counts)), 2) for count in cell_counts
        ]
        unique_cell_proportions = dict(zip(unique_cell_labels, cell_proportions))
        cell_prop_dfs.append(pd.DataFrame([unique_cell_proportions]))

        if include_counts:
            # calculate the area using the tiles containing nuclei to get cells per area
            tile_coords = np.array(db.get_run_state(run_id))
            tile_width = tile_coords[tile_coords[:, 1].argmax() + 1][0]
            tile_height = tile_coords[1][1]
            xs = cell_coords[:, 0]
            ys = cell_coords[:, 1]

            # count how many tiles have at least one cell
            tile_count = 0
            for tile in tile_coords:
                tile_x = tile[0]
                tile_y = tile[1]

                mask = np.logical_and(
                    (np.logical_and(xs >= tile_x, (ys >= tile_y))),
                    (
                        np.logical_and(
                            xs <= (tile_x + tile_width), (ys <= (tile_y + tile_height))
                        )
                    ),
                )
                if np.any(mask):
                    tile_count += 1
            print(f"Number of tiles with a least one cell: {tile_count}")
            tile_area = tile_count * tile_width * tile_height
            slide_pixel_size = db.get_slide_pixel_size_by_evalrun(run_id)
            tile_area = tile_area * slide_pixel_size * slide_pixel_size
            print(f"Tile area in um^2: {tile_area}")

            cell_counts = [count / tile_area * 1000000 for count in cell_counts]
            all_cell_counts = dict(zip(unique_cell_labels, cell_counts))
            cell_counts_df.append(pd.DataFrame([all_cell_counts]))

        # print tissue predictions from tsv file
        pretrained_path = (
            project_dir
            / "results"
            / "graph"
            / model_type
            / exp_name
            / model_weights_dir
            / "eval"
            / model_name
            / f"run_{run_id}"
        )
        tissue_df = pd.read_csv(pretrained_path / "tissue_preds.tsv", sep="\t")
        # remove rows where knots were removed
        if group_knts and not trained_with_grouped_knts:
            tissue_df = tissue_df.loc[
                ~tissue_df.index.isin(inds_to_remove)
            ].reset_index(drop=True)

        tissue_names_mapping = {tissue.label: tissue.name for tissue in organ.tissues}
        tissue_df["class"] = tissue_df["class"].map(tissue_names_mapping)
        unique_tissues, tissue_counts = np.unique(
            tissue_df["class"], return_counts=True
        )
        tissue_proportions = [
            round((count / sum(tissue_counts)), 2) for count in tissue_counts
        ]
        unique_tissue_proportions = dict(zip(unique_tissues, tissue_proportions))
        tissue_prop_dfs.append(pd.DataFrame([unique_tissue_proportions]))

        if include_counts:
            tissue_counts = [count / tile_area * 1000000 for count in tissue_counts]
            all_tissue_counts = dict(zip(unique_tissues, tissue_counts))
            tissue_counts_df.append(pd.DataFrame([all_tissue_counts]))

    cell_df = pd.concat(cell_prop_dfs)
    args_to_sort = np.argsort([cell.structural_id for cell in organ.cells])
    cell_df = cell_df[cell_df.columns[args_to_sort]]
    cell_colours = {cell.name: cell.colour for cell in organ.cells}

    if include_counts:
        cell_counts_df = pd.concat(cell_counts_df)
        cell_counts_df = cell_counts_df[cell_counts_df.columns[args_to_sort]]
        cell_counts_df["Total"] = cell_counts_df[list(cell_counts_df.columns)].sum(
            axis=1
        )

    tissue_df = pd.concat(tissue_prop_dfs)
    tissue_df = tissue_df[
        [
            "Terminal Villi",
            "Mature Intermediate Villi",
            "Stem Villi",
            "Villus Sprout",
            "Anchoring Villi",
            "Chorionic Plate",
            "Basal Plate/Septum",
            "Fibrin",
            "Avascular Villi",
        ]
    ]
    tissue_colours = {tissue.name: tissue.colour for tissue in organ.tissues}

    if include_counts:
        tissue_counts_df = pd.concat(tissue_counts_df)
        tissue_df = tissue_df[
            [
                "Terminal Villi",
                "Mature Intermediate Villi",
                "Stem Villi",
                "Villus Sprout",
                "Anchoring Villi",
                "Chorionic Plate",
                "Basal Plate/Septum",
                "Fibrin",
                "Avascular Villi",
            ]
        ]
        tissue_counts_df["Total"] = tissue_counts_df[
            list(tissue_counts_df.columns)
        ].sum(axis=1)
        tissue_colours["Total"] = "#000000"

    plot_box_and_whisker(cell_df, "plots/cell_proportions.png", "Cell", cell_colours)
    plot_box_and_whisker(cell_counts_df, "plots/cell_counts.png", "Cell", cell_colours)

    plot_box_and_whisker(
        tissue_df, "plots/tissue_proportions.png", "Tissue", tissue_colours
    )
    plot_box_and_whisker(
        tissue_counts_df, "plots/tissue_counts.png", "Tissue", tissue_colours
    )


def plot_box_and_whisker(df, save_path, entity, colours, box=False, swarm=False):
    sns.set_style("white")
    plt.subplots(figsize=(8, 8), dpi=400)
    if swarm:
        if box:
            sns.boxplot(data=df, palette=colours, whis=[0, 100])
            ax = sns.swarmplot(data=df, color=".25")
        else:
            ax = sns.swarmplot(data=df, palette=colours)
    else:
        ax = sns.swarmplot(data=df, color=".5")
        _offset_swarm(ax, 0.3)

        melted_df = pd.melt(df.reset_index(drop=True).T.reset_index(), id_vars="index")
        ax = sns.lineplot(
            data=melted_df,
            x="index",
            y="value",
            hue="index",
            marker="o",
            err_style="bars",
            palette=colours,
            markersize=15,
            legend=False,
        )
        ax.lines[0].set_linestyle("")

    plt.ylim(top=0.62)
    ax.set(ylabel=f"Proportion of {entity}s Across WSIs")
    ax.set(xlabel=f"{entity} Labels")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.clf()


def _offset_swarm(ax, offset):
    path_collections = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.PathCollection)
    ]
    for path_collection in path_collections:
        x, y = np.array(path_collection.get_offsets()).T
        xnew = x + offset
        offsets = list(zip(xnew, y))
        path_collection.set_offsets(offsets)


if __name__ == "__main__":
    typer.run(main)
