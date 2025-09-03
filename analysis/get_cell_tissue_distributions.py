from typing import List, Optional

import typer
import numpy as np
import pandas as pd

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.graph.graph_creation.get_and_process import get_hdf5_data


def main(
    run_ids: Optional[List[int]] = typer.Option([]),
    file_run_ids: Optional[str] = None,
    project_name: str = typer.Option(...),
    group_knts: bool = True,
    include_counts: bool = False,
    include_tissues: bool = True,
):
    """Saves the distribution of cell and tissue types across multiple WSIs as csvs
    Note: if you do not want to group knts then you cannot use the tissue hdf5 file.
    """

    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.name for cell in organ.cells}
    tissue_label_mapping = {tissue.id: tissue.name for tissue in organ.tissues}
    project_dir = get_project_dir(project_name)

    if file_run_ids is not None:
        run_ids = (
            pd.read_csv(project_dir / file_run_ids, header=None)
            .values.flatten()
            .tolist()
        )

    cell_prop_dfs = []
    cell_counts_df = []
    tissue_prop_dfs = []
    tissue_counts_df = []
    for run_id in run_ids:
        # Get path to embeddings hdf5 files
        if not group_knts:
            assert not include_tissues
        hdf5_data = get_hdf5_data(
            project_name, run_id, 0, 0, -1, -1, tissue=include_tissues
        )

        # print cell predictions from hdf5 file
        unique_cells, cell_counts = np.unique(
            hdf5_data.cell_predictions, return_counts=True
        )
        unique_cell_labels = []
        for label in unique_cells:
            unique_cell_labels.append(cell_label_mapping[label])
        cell_proportions = [
            round((count / sum(cell_counts)), 3) for count in cell_counts
        ]
        unique_cell_proportions = dict(zip(unique_cell_labels, cell_proportions))
        cell_prop_dfs.append(pd.DataFrame([unique_cell_proportions]))

        if include_counts:
            # calculate the area using the tiles containing nuclei to get cells per area
            tile_coords = np.array(db.get_run_state(run_id))
            tile_width = tile_coords[tile_coords[:, 1].argmax() + 1][0]
            tile_height = tile_coords[1][1]
            xs = hdf5_data.coords[:, 0]
            ys = hdf5_data.coords[:, 1]

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

        if include_tissues:
            unique_tissues, tissue_counts = np.unique(
                hdf5_data.tissue_predictions, return_counts=True
            )
            unique_tissue_labels = []
            for label in unique_tissues:
                unique_tissue_labels.append(tissue_label_mapping[label])
            tissue_proportions = [
                round((count / sum(tissue_counts)), 3) for count in tissue_counts
            ]
            unique_tissue_proportions = dict(
                zip(unique_tissue_labels, tissue_proportions)
            )
            tissue_prop_dfs.append(pd.DataFrame([unique_tissue_proportions]))

            if include_counts:
                tissue_counts = [count / tile_area * 1000000 for count in tissue_counts]
                all_tissue_counts = dict(zip(unique_tissue_labels, tissue_counts))
                tissue_counts_df.append(pd.DataFrame([all_tissue_counts]))

    cell_df = _reorder_cell_columns(pd.concat(cell_prop_dfs), organ)
    cell_colours = {cell.name: cell.colour for cell in organ.cells}
    cell_colours["Total"] = "#000000"
    cell_df.to_csv("plots/cell_proportions.csv")

    if include_tissues:
        tissue_df = _reorder_tissue_columns(pd.concat(tissue_prop_dfs))
        tissue_colours = {tissue.name: tissue.colour for tissue in organ.tissues}
        tissue_df.to_csv("plots/tissue_proportions.csv")

    if include_counts:
        cell_counts_df = _reorder_cell_columns(pd.concat(cell_counts_df), organ)
        cell_counts_df["Total"] = cell_counts_df[list(cell_counts_df.columns)].sum(
            axis=1
        )
        cell_counts_df.to_csv("plots/cell_counts.csv")

        if include_tissues:
            tissue_counts_df = _reorder_tissue_columns(
                pd.concat(tissue_counts_df)
            )
            tissue_counts_df["Total"] = tissue_counts_df[
                list(tissue_counts_df.columns)
            ].sum(axis=1)
            tissue_colours["Total"] = "#000000"
            tissue_counts_df.to_csv("plots/tissue_counts.csv")


def _reorder_cell_columns(cell_df, organ):
    args_to_sort = np.argsort([cell.structural_id for cell in organ.cells])
    return cell_df[cell_df.columns[args_to_sort]]


def _reorder_tissue_columns(tissue_df):
    return tissue_df[
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


if __name__ == "__main__":
    typer.run(main)
