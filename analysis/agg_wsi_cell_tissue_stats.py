from typing import List, Optional
from pathlib import Path

import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.graph.graph_creation.get_and_process import get_hdf5_data


def main(
    run_ids: Optional[List[int]] = typer.Option([], help="Eval run IDs to aggregate"),
    file_run_ids: Optional[str] = typer.Option(
        None, help="CSV (relative to project dir) of eval run IDs, one per line"
    ),
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option("placenta", help="Organ for cell/tissue definitions"),
    db_name: str = typer.Option("main.db", help="Database file in happy/db/, or an absolute path to a .db file"),
    custom_embeddings_path: Optional[str] = typer.Option(None, help="Custom root path to the project embeddings (overrides default)"),
    tissues: Optional[List[str]] = typer.Option(
        [], help="Restrict to these tissue names (empty = all tissues in the organ)"
    ),
    save_dir: str = typer.Option(
        "results/cell_tissue_distributions",
        help="Directory (relative to project dir) to write outputs to",
    ),
    line_plot: bool = False,
):
    """Aggregate cell-type composition within each tissue type across many WSIs.

    Organ-agnostic: tissue/cell order and colours come from the organ definition.
    Writes a per-run CSV and a mean-across-runs CSV, plus a stacked bar plot of the
    average cell composition within each tissue.
    """
    db.init(db_name)
    organ = get_organ(organ_name)
    cell_label_mapping = {cell.id: cell.name for cell in organ.cells}
    tissue_label_mapping = {tissue.id: tissue.name for tissue in organ.tissues}
    cell_colours = {cell.name: cell.colour for cell in organ.cells}
    cell_order = [c.name for c in sorted(organ.cells, key=lambda c: c.structural_id)]
    tissue_order = [t.name for t in organ.tissues]
    if tissues:
        tissue_order = [t for t in tissue_order if t in tissues]
    project_dir = get_project_dir(project_name)

    if file_run_ids is not None:
        run_ids = (
            pd.read_csv(project_dir / file_run_ids, header=None)
            .values.flatten()
            .tolist()
        )

    per_run_dfs = []
    for run_id in run_ids:
        hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1, tissue=True, custom_path=custom_embeddings_path)

        df = pd.DataFrame(
            {
                "Cells": pd.Series(hdf5_data.cell_predictions).map(cell_label_mapping),
                "Tissues": pd.Series(hdf5_data.tissue_predictions).map(tissue_label_mapping),
            }
        )
        # percentage of each cell type within each tissue type
        grouped = df.groupby(["Tissues", "Cells"]).size().reset_index(name="count")
        grouped["prop"] = grouped.groupby("Tissues")["count"].transform(
            lambda x: x * 100 / x.sum()
        )
        prop_df = grouped.pivot_table(index="Tissues", columns="Cells", values="prop")
        prop_df = prop_df.reindex(index=tissue_order).dropna(how="all")
        prop_df.insert(0, "eval_id", run_id)
        per_run_dfs.append(prop_df)

    all_props = pd.concat(per_run_dfs).fillna(0)

    out_dir = project_dir / save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_props.to_csv(out_dir / "cells_in_tissues_per_run.csv")

    # mean cell composition within each tissue, averaged across runs
    avg = (
        all_props.drop(columns="eval_id")
        .groupby(all_props.index)
        .mean()
        .reindex(index=[t for t in tissue_order if t in all_props.index])
    )
    present_cells = [c for c in cell_order if c in avg.columns]
    avg = avg[present_cells]
    avg.to_csv(out_dir / "avg_cells_in_tissues.csv")

    _plot_stacked(avg, [cell_colours[c] for c in present_cells], out_dir, line_plot)


def _plot_stacked(avg_df, colours, save_dir, line_plot):
    sns.set(style="white")
    plt.rcParams["figure.dpi"] = 600
    if line_plot:
        sns.lineplot(data=avg_df)
        out = save_dir / "avg_cells_in_tissues_line.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
        return

    ax = avg_df.plot(
        kind="bar", stacked=True, color=colours, width=0.8, figsize=(8.5, 6)
    )
    ax.set(xlabel=None)
    plt.xticks(range(len(avg_df.index)), list(avg_df.index), rotation=0, size=9)
    plt.yticks([])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],
        loc="upper center", bbox_to_anchor=(0.5, 1.15),
        prop={"size": 9.25}, ncol=4,
    )
    plt.tight_layout()
    sns.despine(left=True)
    out = save_dir / "avg_cells_in_tissues.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
