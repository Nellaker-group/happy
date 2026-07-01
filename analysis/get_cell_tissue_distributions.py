from typing import List, Optional
from pathlib import Path

import typer
import numpy as np
import pandas as pd

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.graph.graph_creation.get_and_process import get_hdf5_data

# every output CSV starts with these identifier columns
ID_COLS = ["slide_name", "slide_id", "eval_id"]


def main(
    run_ids: Optional[List[int]] = typer.Option([], help="Eval run IDs to include"),
    file_run_ids: Optional[str] = typer.Option(
        None, help="CSV (relative to project dir) of eval run IDs, one per line"
    ),
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option("placenta", help="Organ for cell/tissue definitions"),
    db_name: str = typer.Option("main.db", help="Database file in happy/db/, or an absolute path to a .db file"),
    save_dir: str = typer.Option(
        "results/cell_tissue_distributions",
        help="Directory (relative to project dir) to write the CSVs to",
    ),
    group_knts: bool = True,
    include_counts: bool = typer.Option(True, help="Also compute density (counts per mm^2)"),
    include_tissues: bool = True,
):
    """Save the distribution of cell and tissue types across WSIs as CSVs.

    Writes up to four CSVs, each row a single eval run and each prefixed with the
    slide_name, slide_id and eval_id identifier columns:
        cell_proportions.csv, tissue_proportions.csv  (fraction of cells/nodes)
        cell_counts.csv,       tissue_counts.csv       (density, counts per mm^2)

    Organ-agnostic: column order follows the organ's cell/tissue definitions.
    Note: if you do not group knts you cannot use the tissue hdf5 file.
    """
    db.init(db_name)
    organ = get_organ(organ_name)
    cell_label_mapping = {cell.id: cell.name for cell in organ.cells}
    tissue_label_mapping = {tissue.id: tissue.name for tissue in organ.tissues}
    cell_order = [c.name for c in sorted(organ.cells, key=lambda c: c.structural_id)]
    tissue_order = [t.name for t in organ.tissues]
    project_dir = get_project_dir(project_name)

    if file_run_ids is not None:
        run_ids = (
            pd.read_csv(project_dir / file_run_ids, header=None)
            .values.flatten()
            .tolist()
        )

    cell_prop_rows, cell_count_rows = [], []
    tissue_prop_rows, tissue_count_rows = [], []
    for run_id in run_ids:
        if not group_knts:
            assert not include_tissues
        ids = _identifiers(run_id)
        hdf5_data = get_hdf5_data(
            project_name, run_id, 0, 0, -1, -1, tissue=include_tissues
        )

        unique_cells, cell_counts = np.unique(
            hdf5_data.cell_predictions, return_counts=True
        )
        cell_labels = [cell_label_mapping[label] for label in unique_cells]
        cell_proportions = [round(c / sum(cell_counts), 3) for c in cell_counts]
        cell_prop_rows.append({**ids, **dict(zip(cell_labels, cell_proportions))})

        tile_area = None
        if include_counts:
            tile_area = _tile_area_um2(run_id, hdf5_data)
            cell_density = [c / tile_area * 1e6 for c in cell_counts]
            cell_count_rows.append({**ids, **dict(zip(cell_labels, cell_density))})

        if include_tissues:
            unique_tissues, tissue_counts = np.unique(
                hdf5_data.tissue_predictions, return_counts=True
            )
            tissue_labels = [tissue_label_mapping[label] for label in unique_tissues]
            tissue_proportions = [round(c / sum(tissue_counts), 3) for c in tissue_counts]
            tissue_prop_rows.append(
                {**ids, **dict(zip(tissue_labels, tissue_proportions))}
            )

            if include_counts:
                tissue_density = [c / tile_area * 1e6 for c in tissue_counts]
                tissue_count_rows.append(
                    {**ids, **dict(zip(tissue_labels, tissue_density))}
                )

    out_dir = project_dir / save_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _save(cell_prop_rows, cell_order, out_dir / "cell_proportions.csv")
    if include_counts:
        _save(cell_count_rows, cell_order, out_dir / "cell_counts.csv", total=True)
    if include_tissues:
        _save(tissue_prop_rows, tissue_order, out_dir / "tissue_proportions.csv")
        if include_counts:
            _save(
                tissue_count_rows, tissue_order, out_dir / "tissue_counts.csv", total=True
            )


def _identifiers(run_id):
    """slide_name, slide_id, eval_id for an eval run, as the leading CSV columns."""
    eval_run = db.get_eval_run_by_id(run_id)
    slide = eval_run.slide
    return {"slide_name": slide.slide_name, "slide_id": slide.id, "eval_id": run_id}


def _tile_area_um2(run_id, hdf5_data):
    """Area (um^2) of tiles containing at least one cell, used for density."""
    tile_coords = np.array(db.get_run_state(run_id))
    tile_width = tile_coords[tile_coords[:, 1].argmax() + 1][0]
    tile_height = tile_coords[1][1]
    xs, ys = hdf5_data.coords[:, 0], hdf5_data.coords[:, 1]

    tile_count = 0
    for tile_x, tile_y in tile_coords:
        mask = (
            (xs >= tile_x)
            & (ys >= tile_y)
            & (xs <= tile_x + tile_width)
            & (ys <= tile_y + tile_height)
        )
        if np.any(mask):
            tile_count += 1
    print(f"run {run_id}: {tile_count} tiles with at least one cell")

    slide_pixel_size = db.get_slide_pixel_size_by_evalrun(run_id)
    return tile_count * tile_width * tile_height * slide_pixel_size * slide_pixel_size


def _save(rows, structure_order, csv_path, total=False):
    """Write rows to CSV: identifier columns first, then structures in organ order."""
    df = pd.DataFrame(rows).fillna(0)
    present = [s for s in structure_order if s in df.columns]
    df = df[ID_COLS + present]
    if total:
        df["Total"] = df[present].sum(axis=1)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    typer.run(main)
