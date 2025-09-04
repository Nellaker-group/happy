from pathlib import Path
from typing import Optional

import typer

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.hdf5 import get_embeddings_file
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from happy.utils.utils import get_project_dir
from happy.graph.utils.visualise_points import visualize_points


def main(
    run_id: int = typer.Option(...),
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    single_cell: Optional[str] = None,
    custom_save_dir: Optional[str] = None,
    cell: bool = True,
    tissue: bool = True,
):
    """Saves cell predictions coloured by cell type across whole slide

    Args:
        run_id: id of the run which generated the embeddings file
        project_name: name of the project
        organ_name: name of the organ to get the cell colours
        single_cell: if specified, only this cell type will be plotted
        custom_save_dir: if specified, the visualisation will be saved to this directory
        cell: whether to plot cell predictions
        tissue: whether to plot tissue predictions
    """
    # Create database connection
    db.init()

    organ = get_organ(organ_name)
    project_dir = get_project_dir(project_name)

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_name, run_id)
    hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1, tissue=tissue)

    if single_cell:
        hdf5_data = hdf5_data.filter_by_cell_type(single_cell, organ)

    # setup save location and filename
    base_save_dir = project_dir / "visualisations" / "graphs"
    save_dir = base_save_dir / custom_save_dir if custom_save_dir else base_save_dir
    lab = Path(embeddings_path.parts[-3])
    save_dir = save_dir / lab
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"{embeddings_path.parts[-2].split('.')[0]}"
    plot_name = f"{plot_name}_{single_cell}" if single_cell else plot_name

    if cell:
        curr_plot_name = f"{plot_name}_cells.png"
        visualize_points(
            organ,
            save_dir / curr_plot_name,
            hdf5_data.coords.astype("int32"),
            labels=hdf5_data.cell_predictions,
            width=-1,
            height=-1,
        )
        print(f"Plot saved to {save_dir / curr_plot_name}")
    if tissue:
        curr_plot_name = f"{plot_name}_tissues.png"
        colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
        colours = [colours_dict[label] for label in hdf5_data.tissue_predictions]
        visualize_points(
            organ,
            save_dir / curr_plot_name,
            hdf5_data.coords.astype("int32"),
            labels=hdf5_data.tissue_predictions,
            colours=colours,
            width=-1,
            height=-1,
        )
        print(f"Plot saved to {save_dir / curr_plot_name}")


if __name__ == "__main__":
    typer.run(main)
