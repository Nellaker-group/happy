from pathlib import Path
from typing import Optional

import typer
import numpy as np
import matplotlib.pyplot as plt

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.utils.utils import get_project_dir
from happy.graph.graph_creation.get_and_process import get_hdf5_data

NUC_COLOUR = "black"


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
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
    he_max_size: int = typer.Option(4000, help="Max H&E thumbnail dimension in px"),
    point_size: float = typer.Option(1.0, help="Scatter point size"),
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
    """
    if not any([he, nuc, cell, tissue]):
        raise typer.BadParameter("Enable at least one of --he / --nuc / --cell / --tissue")
    if (nuc or cell or tissue) and eval_id is None:
        raise typer.BadParameter("--eval-id is required for --nuc/--cell/--tissue")

    db.init()
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

    background = None
    if he or overlay:
        he_img, extent = _load_he(slide_path, he_max_size)
        if he:
            _save_he(he_img, extent, out_dir / f"{stem}_he.png")
        if overlay:
            background = (he_img, extent)

    hdf5 = None
    if cell or tissue:
        hdf5 = get_hdf5_data(project_name, eval_id, 0, 0, -1, -1, tissue=tissue)

    if nuc:
        coords = _nuclei_coords(eval_id)
        _save_points(coords, NUC_COLOUR, out_dir / f"{stem}_nuclei.png", background, point_size)
    if cell:
        colours = _organ_colours(organ.cells, hdf5.cell_predictions)
        _save_points(hdf5.coords, colours, out_dir / f"{stem}_cells.png", background, point_size)
    if tissue:
        colours = _organ_colours(organ.tissues, hdf5.tissue_predictions)
        _save_points(hdf5.coords, colours, out_dir / f"{stem}_tissues.png", background, point_size)


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


def _nuclei_coords(run_id):
    preds = db.get_all_prediction_coordinates(run_id)
    return np.array([[p["x"], p["y"]] for p in preds])


def _organ_colours(structures, predictions):
    colour_map = {s.id: s.colour for s in structures}
    return [colour_map[p] for p in predictions]


def _figsize(width, height, max_dim=15):
    aspect = (height / width) if width else 1.0
    return (max_dim / aspect, max_dim) if aspect >= 1 else (max_dim, max_dim * aspect)


def _save_he(he_img, extent, path):
    fig = plt.figure(figsize=_figsize(extent[1], extent[2]), dpi=300)
    plt.imshow(he_img, extent=extent)
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    print(f"Saved {path}")


def _save_points(coords, colours, path, background, point_size):
    coords = np.asarray(coords)
    if coords.size == 0:
        print(f"No points to plot for {path.name}, skipping")
        return
    xs, ys = coords[:, 0], coords[:, 1]

    if background is not None:
        he_img, extent = background
        fig = plt.figure(figsize=_figsize(extent[1], extent[2]), dpi=300)
        plt.imshow(he_img, extent=extent)
        plt.scatter(xs, ys, marker=".", s=point_size, c=colours, zorder=1000)
        plt.xlim(0, extent[1])
        plt.ylim(extent[2], 0)
    else:
        fig = plt.figure(figsize=_figsize(xs.max() - xs.min(), ys.max() - ys.min()), dpi=300)
        plt.scatter(xs, ys, marker=".", s=point_size, c=colours, zorder=1000)
        plt.gca().invert_yaxis()

    plt.axis("off")
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    typer.run(main)
