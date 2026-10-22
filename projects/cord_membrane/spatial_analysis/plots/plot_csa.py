import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import typer
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter

from happy.utils.utils import get_project_dir
from projects.cord_membrane.spatial_analysis.plots.style import (
    INK,
    MUTED,
    NEUTRAL_FILL,
    STRUCTURE_COLOURS,
    STRUCTURE_NAMES,
    STRUCTURE_SHORT,
    SURFACE,
    style_axes,
)

VESSEL_MEASURES = [
    ("lumen_area_um2", "Lumen area (mm²)", 1e-6),
    ("wall_area_um2", "Wall area (mm²)", 1e-6),
    ("wagenworth_index", "Wagenworth index (log scale)", 1),
    ("wall_thickness_um", "Wall thickness (µm)", 1),
]


def main(
    run_id: int = typer.Option(...),
    project_name: str = "cord_membrane",
):
    """Plot the cord sections and vessel measurements from run_csa for one run.

    Saves two figures next to the CSA results:
        run_N_outlines.png: each cord section with its vessels and any remnants,
            labelled as in the CSVs (A1 artery 1, V1 vein 1), and their lumens
        run_N_vessels.png: lumen area, wall area, Wagenworth index and wall
            thickness of each vessel

    Run after run_csa:
        python -m projects.cord_membrane.spatial_analysis.plots.plot_csa --run-id 1

    Args:
        run_id: eval run the CSA was run on
        project_name: project dir containing the results
    """
    results_dir = get_project_dir(project_name) / "results" / "csa"
    outlines = _read_outlines(results_dir / f"run_{run_id}_outlines.geojson")
    sections = pd.read_csv(results_dir / f"run_{run_id}_sections.csv")
    vessels = pd.read_csv(results_dir / f"run_{run_id}_vessels.csv")

    _plot_outlines(outlines, sections, run_id, results_dir / f"run_{run_id}_outlines.png")
    if len(vessels):
        _plot_vessels(vessels, run_id, results_dir / f"run_{run_id}_vessels.png")


def _read_outlines(path):
    # feature names are "Section <section> <tissue> <index>", with " lumen" for lumens
    outlines = []
    with open(path) as f:
        for feature in json.load(f)["features"]:
            _, section, tissue, index, *lumen = feature["properties"]["name"].split()
            outlines.append(
                {
                    "section": int(section),
                    "tissue": tissue,
                    "index": int(index),
                    "lumen": bool(lumen),
                    "shape": shapely.geometry.shape(feature["geometry"]),
                }
            )
    return outlines


def _plot_outlines(outlines, sections, run_id, save_path):
    section_ids = sorted({o["section"] for o in outlines})
    ncols = min(3, len(section_ids))
    nrows = math.ceil(len(section_ids) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows + 0.6), squeeze=False, layout="constrained"
    )

    # the same span in every panel, so sections can be compared by eye
    section_shapes = {o["section"]: o["shape"] for o in outlines if o["tissue"] == "Section"}
    span = max(max(s.bounds[2] - s.bounds[0], s.bounds[3] - s.bounds[1]) for s in section_shapes.values())
    span *= 1.05

    present = set()
    for ax, section_id in zip(axes.flat, section_ids):
        items = [o for o in outlines if o["section"] == section_id]
        _fill(ax, section_shapes[section_id], NEUTRAL_FILL)
        for o in items:
            if o["tissue"] in STRUCTURE_COLOURS and not o["lumen"]:
                _fill(ax, o["shape"], STRUCTURE_COLOURS[o["tissue"]], edge=SURFACE)
                present.add(o["tissue"])
        for o in items:
            if o["lumen"]:
                _fill(ax, o["shape"], SURFACE, edge=MUTED)
        lumens = {(o["tissue"], o["index"]): o["shape"] for o in items if o["lumen"]}
        for o in items:
            if o["tissue"] in STRUCTURE_COLOURS and not o["lumen"]:
                # on the structure's own wall, outside its lumen
                wall = o["shape"].difference(lumens.get((o["tissue"], o["index"]), shapely.Polygon()))
                point = (wall if not wall.is_empty else o["shape"]).representative_point()
                ax.text(
                    point.x,
                    point.y,
                    f"{STRUCTURE_SHORT[o['tissue']]}{o['index']}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8},
                )

        cx, cy = section_shapes[section_id].centroid.coords[0]
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy + span / 2, cy - span / 2)
        ax.set_aspect("equal")
        ax.axis("off")
        area = sections.loc[sections["section"] == section_id, "area_um2"].iloc[0] / 1e6
        ax.set_title(f"Section {section_id} · {area:.1f} mm²", fontsize=9, color=INK)
    for ax in axes.flat[len(section_ids):]:
        ax.axis("off")

    _add_scale_bar(axes.flat[0], sections, section_shapes)
    order = [t for t in STRUCTURE_COLOURS if t in present]
    handles = [Patch(facecolor=STRUCTURE_COLOURS[t], label=STRUCTURE_NAMES[t]) for t in order]
    handles += [
        Patch(facecolor=SURFACE, edgecolor=MUTED, label="Lumen"),
        Patch(facecolor=NEUTRAL_FILL, label="Wharton's jelly and other tissue"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"Cord sections, run {run_id}", fontsize=10, color=INK)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


def _fill(ax, shape, colour, edge=None):
    for part in shapely.get_parts(shape):
        if part.is_empty or part.geom_type != "Polygon":
            continue
        ax.add_patch(
            Polygon(
                np.asarray(part.exterior.coords),
                closed=True,
                facecolor=colour,
                edgecolor=edge or colour,
                linewidth=0.6,
            )
        )


def _add_scale_bar(ax, sections, section_shapes):
    # pixel size from a section's area in um2 and in pixels
    section_id = min(section_shapes)
    area_um2 = sections.loc[sections["section"] == section_id, "area_um2"].iloc[0]
    pixel_size = math.sqrt(area_um2 / section_shapes[section_id].area)
    length = 1000 / pixel_size
    x0, x1 = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    x = x0 + 0.05 * (x1 - x0)
    y = y_bottom - 0.05 * (y_bottom - y_top)
    ax.plot([x, x + length], [y, y], color=INK, linewidth=2, solid_capstyle="butt")
    ax.text(x + length / 2, y - 0.02 * (y_bottom - y_top), "1 mm", ha="center", va="bottom", fontsize=8, color=INK)


def _plot_vessels(vessels, run_id, save_path):
    vessels = vessels.sort_values(["section", "tissue", "index"]).reset_index(drop=True)
    labels = [
        f"S{s} {STRUCTURE_SHORT[t]}{i}"
        for s, t, i in zip(vessels["section"], vessels["tissue"], vessels["index"])
    ]
    # a gap between sections
    x = np.arange(len(vessels)) + vessels["section"].rank(method="dense").to_numpy() - 1
    colours = [STRUCTURE_COLOURS[t] for t in vessels["tissue"]]

    fig, axes = plt.subplots(
        2, 2, figsize=(max(8, 0.32 * x.max() + 3), 6.5), sharex=True, layout="constrained"
    )
    for ax, (column, title, scale) in zip(axes.flat, VESSEL_MEASURES):
        values = vessels[column].to_numpy(dtype=float) * scale
        values[values <= 0] = np.nan
        ax.bar(x, values, width=0.7, color=colours)
        if column == "wagenworth_index":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_title(title, fontsize=9, color=INK)
        style_axes(ax)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)

    present = [t for t in STRUCTURE_COLOURS if t in set(vessels["tissue"])]
    handles = [Patch(facecolor=STRUCTURE_COLOURS[t], label=STRUCTURE_NAMES[t]) for t in present]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"Vessel measurements, run {run_id}", fontsize=10, color=INK)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    typer.run(main)
