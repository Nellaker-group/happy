from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import h5py
import typer
import numpy as np
import matplotlib.pyplot as plt

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.utils.utils import get_project_dir
from happy.graph.graph_creation.get_and_process import get_hdf5_data

NUC_COLOUR = "black"
TITLES = {"he": "H&E", "nuclei": "Nuclei", "cells": "Cells", "tissues": "Tissues"}


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    db_name: str = typer.Option("main.db", help="Database file in happy/db/, or an absolute path to a .db file"),
    custom_embeddings_path: Optional[str] = typer.Option(None, help="Custom root path to the project embeddings (overrides default)"),
    hdf5_file: Optional[str] = typer.Option(
        None, help="HDF5 file to plot cell/tissue predictions from instead of the run's embeddings"
    ),
    eval_id: Optional[int] = typer.Option(
        None, help="Eval run id to plot (required for --nuc/--cell/--tissue)"
    ),
    slide_id: Optional[int] = typer.Option(
        None, help="Slide id, for an H&E-only plot when there is no eval run"
    ),
    he: bool = typer.Option(False, help="Render the H&E thumbnail"),
    nuc: bool = typer.Option(False, help="Plot nuclei predictions (black)"),
    cell: bool = typer.Option(False, help="Plot cell predictions (organ colours)"),
    tissue: bool = typer.Option(False, help="Plot tissue predictions (organ colours)"),
    overlay: bool = typer.Option(False, help="Plot predictions over the H&E"),
    one_plot: bool = typer.Option(False, help="Save all enabled layers side by side in one figure"),
    he_max_size: int = typer.Option(4000, help="Max H&E thumbnail dimension in px"),
    point_size: float = typer.Option(0.8, help="Scatter point size"),
    dpi: int = typer.Option(600, help="Resolution of the saved figures"),
    save_dir: str = typer.Option(
        "visualisations/predictions", help="Dir (relative to project) to save to"
    ),
):
    """Visualise nuclei / cell / tissue predictions for one slide, optionally over H&E.

    Organ-agnostic: 
    cell and tissue points use the organ's colours, nuclei are black.
    Pass --eval-id to plot predictions 
    pass just --slide-id with --he for an H&E image when no eval run exists 
    with --overlay, enabled prediction layers are drawn on top of the H&E thumbnail
    with --one-plot, all enabled layers are saved side by side in one figure
    """
    if not any([he, nuc, cell, tissue]):
        raise typer.BadParameter("Enable at least one of --he / --nuc / --cell / --tissue")
    if (nuc or cell or tissue) and eval_id is None:
        raise typer.BadParameter("--eval-id is required for --nuc/--cell/--tissue")

    db.init(db_name)
    organ = get_organ(organ_name)
    project_dir = get_project_dir(project_name)

    if eval_id is not None:
        slide_db_id = db.get_eval_run_by_id(eval_id).slide.id
    elif slide_id is not None:
        slide_db_id = slide_id
    else:
        raise typer.BadParameter("Provide --eval-id or --slide-id")
    slide_path = db.get_slide_path_by_id(slide_db_id)

    out_dir = project_dir / save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"slide{slide_db_id}" + (f"_run{eval_id}" if eval_id is not None else "")

    he_image = _load_he(slide_path, he_max_size) if he or overlay else None

    hdf5 = None
    if (cell or tissue) and hdf5_file:
        hdf5 = _load_predictions_file(hdf5_file)
    elif cell or tissue:
        hdf5 = get_hdf5_data(project_name, eval_id, 0, 0, -1, -1, tissue=tissue, custom_path=custom_embeddings_path)

    # (name, coords, colours) for each enabled layer
    panels = [("he", None, None)] if he else []
    if nuc:
        panels.append(("nuclei", _nuclei_coords(eval_id), NUC_COLOUR))
    if cell:
        panels.append(("cells", hdf5.coords, _organ_colours(organ.cells, hdf5.cell_predictions)))
    if tissue:
        panels.append(("tissues", hdf5.coords, _organ_colours(organ.tissues, hdf5.tissue_predictions)))
    for name, coords, _ in panels:
        if name != "he" and len(coords) == 0:
            print(f"No points to plot for {name}, skipping")
    panels = [p for p in panels if p[0] == "he" or len(p[1]) > 0]
    if not panels:
        return

    if one_plot:
        if he_image is not None:
            limits = he_image[1]
        else:
            limits = _bounds(np.concatenate([np.asarray(p[1]) for p in panels]))
        path = out_dir / f"{stem}_{'_'.join(p[0] for p in panels)}.png"
        _save_panels(panels, he_image, overlay, point_size, limits, path, dpi)
    else:
        for panel in panels:
            name, coords, _ = panel
            limits = he_image[1] if name == "he" or overlay else _bounds(coords)
            _save_panels([panel], he_image, overlay, point_size, limits, out_dir / f"{stem}_{name}.png", dpi)


def _load_he(slide_path, max_size):
    """Return (RGB thumbnail array, matplotlib extent) for the whole slide.

    extent is in level-0 WSI pixel coords so prediction coordinates overlay directly.
    """
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.dimensions
    thumb = np.asarray(slide.get_thumbnail((max_size, max_size)).convert("RGB"))
    slide.close()
    return thumb, (0, w, h, 0)


def _load_predictions_file(path):
    """Coordinates and cell / tissue predictions from any HDF5 file that has them."""
    with h5py.File(path, "r") as f:
        cell_key = "cell_predictions" if "cell_predictions" in f else "predictions"
        return SimpleNamespace(
            coords=f["coords"][:],
            cell_predictions=f[cell_key][:] if cell_key in f else None,
            tissue_predictions=f["tissue_predictions"][:] if "tissue_predictions" in f else None,
        )


def _nuclei_coords(run_id):
    preds = db.get_all_prediction_coordinates(run_id)
    return np.array([[p["x"], p["y"]] for p in preds])


def _organ_colours(structures, predictions):
    colour_map = {s.id: s.colour for s in structures}
    return [colour_map[p] for p in predictions]


def _figsize(width, height, max_dim=15):
    aspect = (height / width) if width else 1.0
    return (max_dim / aspect, max_dim) if aspect >= 1 else (max_dim, max_dim * aspect)


def _bounds(coords):
    """(left, right, bottom, top) of the points, with y pointing down like the H&E."""
    coords = np.asarray(coords)
    return (coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].max(), coords[:, 1].min())


def _save_panels(panels, he_image, overlay, point_size, limits, path, dpi):
    """Save one or more layers side by side, all drawn within the same limits."""
    width, height = limits[1] - limits[0], limits[2] - limits[3]
    panel_width, panel_height = _figsize(width, height, max_dim=8 if len(panels) > 1 else 15)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(panel_width * len(panels), panel_height), dpi=dpi, squeeze=False
    )
    for ax, (name, coords, colours) in zip(axes[0], panels):
        if name == "he" or overlay:
            ax.imshow(he_image[0], extent=he_image[1])
        if name != "he":
            coords = np.asarray(coords)
            ax.scatter(
                coords[:, 0], coords[:, 1], marker="o", s=point_size, c=colours, linewidths=0, zorder=1000
            )
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.axis("off")
        if len(panels) > 1:
            ax.set_title(TITLES[name])
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    typer.run(main)
