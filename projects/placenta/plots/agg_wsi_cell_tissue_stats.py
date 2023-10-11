from typing import List
from pathlib import Path

import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.utils.hdf5 import get_datasets_in_patch, get_embeddings_file
from happy.graph.process_knts import process_knt_cells


def main(
    run_ids: List[int] = typer.Option(...),
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: str = "sup_clustergcn",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    group_knts: bool = False,
    trained_with_grouped_knts: bool = False,
    villus_only: bool = True,
    line_plot: bool = False,
):
    """Aggregated cells within and tissues for all given WSIs."""

    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.label for cell in organ.cells}
    cell_colours = {cell.label: cell.colour for cell in organ.cells}
    project_dir = get_project_dir(project_name)

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
    )

    all_prop_dfs = []
    for run_id in run_ids:
        # Get path to embeddings hdf5 files
        embeddings_path = get_embeddings_file(project_dir, run_id)
        # Get hdf5 datasets contained in specified box/patch of WSI
        predictions, embeddings, cell_coords, confidence = get_datasets_in_patch(
            embeddings_path, x_min, y_min, width, height
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
        tissue_pred_path = pretrained_path / f"run_{run_id}"
        tissue_df = pd.read_csv(
            tissue_pred_path / "tissue_preds.tsv", sep="\t", names=["x", "y", "Tissues"]
        )
        # remove rows where knots were removed
        if group_knts and not trained_with_grouped_knts:
            tissue_df = tissue_df.loc[
                ~tissue_df.index.isin(inds_to_remove)
            ].reset_index(drop=True)

        # get number of cell types within each tissue type
        cell_df = pd.DataFrame(
            {"x": cell_coords[:, 0], "y": cell_coords[:, 1], "Cells": predictions}
        )
        cell_df["Cells"] = cell_df["Cells"].map(cell_label_mapping)
        cell_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)
        tissue_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)

        combined_df = pd.merge(cell_df, tissue_df)

        grouped_df = (
            combined_df.groupby(["Tissues", "Cells"]).size().reset_index(name="count")
        )
        grouped_df["prop"] = grouped_df.groupby(["Tissues"])["count"].transform(
            lambda x: x * 100 / x.sum()
        )
        prop_df = grouped_df.pivot_table(
            index="Tissues", columns="Cells", values="prop"
        )
        prop_df = prop_df[reversed(prop_df.columns)]

        if villus_only:
            prop_df = prop_df.drop(
                ["Fibrin", "Avascular", "Maternal", "AVilli"], axis=0
            )
            prop_df = prop_df.reindex(
                ["Chorion", "SVilli", "MIVilli", "TVilli", "Sprout"]
            )
        all_prop_dfs.append(prop_df)
    all_props = pd.concat(all_prop_dfs)
    all_props = all_props.fillna(0)

    plot_save_path = Path(*pretrained_path.parts[:-1])
    if line_plot:
        sns.lineplot(data=all_props, palette=cell_colours)
        plt.savefig(plot_save_path / "all_cells_in_tissues_line.png")
    else:
        # Reorder the stacked cells for plotting clarity
        all_props = all_props[
            [
                "MES",
                "MAT",
                "EVT",
                "KNT",
                "HOF",
                "WBC",
                "FIB",
                "VMY",
                "VEN",
                "CYT",
                "SYN",
            ]
        ]
        avg_cells_within_tissues(organ, all_props, cell_colours, plot_save_path)


def avg_cells_within_tissues(organ, prop_df, cell_colours_mapping, save_path):
    tissue_label_to_name = {tissue.label: tissue.name for tissue in organ.tissues}
    cell_label_to_name = {cell.label: cell.name for cell in organ.cells}
    avg_prop_df = prop_df.groupby(prop_df.index, axis=0).mean()
    avg_prop_df = avg_prop_df.reindex(
        ["Chorion", "SVilli", "MIVilli", "TVilli", "Sprout"]
    )

    avg_prop_df.index = avg_prop_df.index.map(tissue_label_to_name)
    cell_colours = [cell_colours_mapping[cell] for cell in avg_prop_df.columns]
    avg_prop_df.columns = avg_prop_df.columns.map(cell_label_to_name)

    sns.set(style="white")
    plt.rcParams["figure.dpi"] = 600
    ax = avg_prop_df.plot(
        kind="bar",
        stacked=True,
        color=cell_colours,
        legend="reverse",
        width=0.8,
        figsize=(8.5, 6),
    )
    ax.set(xlabel=None)
    plt.xticks(
        range(len(avg_prop_df.index)), list(avg_prop_df.index), rotation=0, size=9
    )
    plt.yticks([])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        prop={"size": 9.25},
        ncol=4,
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.tight_layout()
    sns.despine(left=True)
    plt.savefig(save_path / "avg_all_cells_in_tissues.png")
    print(f"Avg plot saved to {save_path / 'cells_in_tissues.png'}")
    plt.close()
    plt.clf()


if __name__ == "__main__":
    typer.run(main)
