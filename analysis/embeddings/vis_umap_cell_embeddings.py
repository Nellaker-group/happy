import time

import typer
import umap
import umap.plot
from bokeh.plotting import show, save

import happy.db.eval_runs_interface as db
from happy.hdf5 import get_embeddings_file, HDF5Dataset
from happy.organs import get_organ
from utils import setup, embeddings_results_path
from plots import plot_interactive, plot_umap


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    subset_start: float = 0.0,
    num_points: int = -1,
    interactive: bool = False,
    use_tissue_hdf5: bool = True,
    use_tissue_embeddings: bool = False,
    plot_cell: bool = True,
    plot_tissue: bool = False,
):
    """Plots and saves a UMAP from the cell embedding vectors.

    Can be an interactive html file (open with browser or IDE) or simply a png.

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project dir to save to
        run_id: id of the run which created the UMAP embeddings
        subset_start: at which index or proportion of the file to start (int or float)
        num_points: number of points to include in the UMAP from subset_start onwards
        interactive: to save an interactive html or a png
        use_tissue_hdf5: whether to use the hdf5 file which includes tissues
        use_tissue_embeddings: whether to use the tissue embeddings
        plot_cell: whether to plot cell predictions
        plot_tissue: whether to plot tissue predictions
    """
    assert plot_cell or plot_tissue, "At least one of cell or tissue must be True"
    embedding_type = "tissue" if use_tissue_embeddings else "cell"

    db.init()
    timer_start = time.time()
    lab_id, slide_name = setup(db, run_id)
    organ = get_organ(organ_name)

    embeddings_file = get_embeddings_file(project_name, run_id, tissue=use_tissue_hdf5)
    save_dir = embeddings_results_path(embeddings_file, lab_id, slide_name, run_id)
    hdf5_data = HDF5Dataset.from_path(
        embeddings_file, start=subset_start, num_points=num_points
    )
    if use_tissue_embeddings:
        embeddings = hdf5_data.tissue_embeddings
    else:
        embeddings = hdf5_data.cell_embeddings

    # Generate UMAP
    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.0, n_neighbors=30)
    mapper = reducer.fit(embeddings)

    if plot_cell:
        if interactive:
            _plot_interactive(
                save_dir, "cell", embedding_type, slide_name, organ, hdf5_data, mapper
            )
        else:
            _plot_image(save_dir, "cell", embedding_type, organ, hdf5_data, mapper)
    if plot_tissue:
        if interactive:
            _plot_interactive(
                save_dir, "tissue", embedding_type, slide_name, organ, hdf5_data, mapper
            )
        else:
            _plot_image(save_dir, "tissue", embedding_type, organ, hdf5_data, mapper)

    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")


def _plot_interactive(
    save_dir, plot_type, embedding_type, slide_name, organ, hdf5_data, mapper
):
    plot_name = (
        f"{hdf5_data.start}-{hdf5_data.end}_{plot_type}_on_{embedding_type}.html"
    )
    coords = hdf5_data.coords
    if plot_type == "tissue":
        predictions = hdf5_data.tissue_predictions
        confidence = hdf5_data.tissue_confidence
    else:
        predictions = hdf5_data.cell_predictions
        confidence = hdf5_data.cell_confidence
    plot = plot_interactive(
        plot_name, slide_name, organ, predictions, confidence, coords, mapper
    )
    show(plot)
    print(f"saving interactive to {save_dir / plot_name}")
    save(plot, save_dir / plot_name)


def _plot_image(save_dir, plot_type, embedding_type, organ, hdf5_data, mapper):
    plot_name = f"{hdf5_data.start}-{hdf5_data.end}_{plot_type}_on_{embedding_type}.png"
    if plot_type == "tissue":
        predictions = hdf5_data.tissue_predictions
        plot = plot_umap(organ, predictions, mapper, tissue=True)
    else:
        predictions = hdf5_data.cell_predictions
        plot = plot_umap(organ, predictions, mapper, tissue=False)
    print(f"saving plot to {save_dir / plot_name}")
    plot.figure.savefig(save_dir / plot_name)


if __name__ == "__main__":
    typer.run(main)
