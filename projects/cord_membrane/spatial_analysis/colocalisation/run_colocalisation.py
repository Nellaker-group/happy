from typing import List, Optional

import numpy as np
import pandas as pd
import typer

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.utils.utils import get_project_dir
from projects.cord_membrane.postprocessing.reclassed_file import load_predictions
from projects.cord_membrane.spatial_analysis.colocalisation.ripley import (
    cross_k,
    random_scatter_k,
)
from projects.cord_membrane.spatial_analysis.geometry import alpha_triangles

# Cell types that leukocytes are compared against, by organ
TARGET_CELLS = {
    "placenta_cord": ["VMY", "MES", "FIB", "MFB", "EPI"],
    "placenta_membrane": ["EVT", "MAT", "MES", "FIB", "EPI"],
}


def main(
    run_id: int = typer.Option(...),
    organ_name: str = typer.Option(..., help="placenta_cord or placenta_membrane"),
    project_name: str = "cord_membrane",
    db_name: str = "main.db",
    custom_path: Optional[str] = None,
    reclassed: bool = True,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    targets: List[str] = typer.Option([]),
    tissue: Optional[str] = None,
    max_distance_um: float = 100.0,
    step_um: float = 1.0,
    summary_distance_um: float = 50.0,
    num_simulations: int = 39,
    alpha_radius_um: float = 50.0,
    seed: int = 0,
):
    """Colocalisation of leukocytes with other cell types using Ripley's cross-K.

    For each target cell type, K(d) counts the leukocytes within d of each target
    cell, scaled by leukocyte density. It is compared with K when the leukocytes
    are scattered at random over the same tissue, repeated num_simulations times
    (median and 2.5-97.5% range), so edge effects apply to both. k_ratio is the
    observed K over the median random K: above 1 leukocytes cluster around that
    cell type, below 1 they avoid it. The tissue excludes empty space such as
    gaps between roll layers and lumens.

    Saves a curve over distances up to max_distance_um and a summary at
    summary_distance_um for each target cell type.

    Run after postprocessing.run_postprocessing (or after tissue inference with
    --no-reclassed):
        python -m projects.cord_membrane.spatial_analysis.colocalisation.run_colocalisation \\
            --run-id 1 --organ-name placenta_membrane

    Args:
        run_id: eval run with cell and tissue predictions
        organ_name: placenta_cord or placenta_membrane (membrane rolls)
        project_name: project dir containing the embeddings and results
        db_name: database containing the eval run
        custom_path: project dir to read embeddings from, if not project_name
        reclassed: use the reclassified predictions from run_postprocessing,
            otherwise the original predictions
        x_min: bottom left x coordinate of the region to analyse (0 for full WSI)
        y_min: bottom left y coordinate of the region to analyse (0 for full WSI)
        width: width of the region to analyse (-1 for full WSI)
        height: height of the region to analyse (-1 for full WSI)
        targets: target cell labels, defaults to TARGET_CELLS for the organ
        tissue: only use cells in this tissue label (e.g. Decidua), whole section if not given
        max_distance_um: largest distance on the K curve
        step_um: spacing of distances on the K curve
        summary_distance_um: distance for the per cell type summary
        num_simulations: number of random scatterings of the leukocytes
        alpha_radius_um: alpha shape radius for the tissue; empty regions wider
            than about twice this are left out
        seed: random seed for the simulations
    """
    if organ_name not in TARGET_CELLS:
        raise typer.BadParameter(f"organ_name must be one of {list(TARGET_CELLS)}")
    db.init(db_name)
    organ = get_organ(organ_name)
    pixel_size = db.get_slide_pixel_size_by_evalrun(run_id)
    targets = targets or TARGET_CELLS[organ_name]
    save_dir = get_project_dir(project_name) / "results" / "colocalisation"
    save_dir.mkdir(parents=True, exist_ok=True)

    coords, cell_predictions, tissue_predictions = load_predictions(
        project_name,
        run_id,
        organ_name,
        reclassed,
        x_min,
        y_min,
        width,
        height,
        custom_path,
    )
    coords_um = coords * pixel_size
    if tissue is not None:
        in_tissue = tissue_predictions == organ.tissue_by_label(tissue).id
        coords_um, cell_predictions = coords_um[in_tissue], cell_predictions[in_tissue]
    triangles, areas = alpha_triangles(coords_um, alpha_radius_um)
    area_um2 = areas.sum()
    leukocytes = coords_um[cell_predictions == organ.cell_by_label("WBC").id]
    print(f"{len(leukocytes)} leukocytes in {area_um2 / 1e6:.2f} mm2 of {tissue or 'tissue'}")

    distances_um = np.arange(step_um, max_distance_um + step_um / 2, step_um)
    radii = np.append(distances_um, summary_distance_um)
    rng = np.random.default_rng(seed)
    curves, summaries = [], []
    for target in targets:
        target_points = coords_um[cell_predictions == organ.cell_by_label(target).id]
        observed = cross_k(leukocytes, target_points, area_um2, radii)
        simulated = random_scatter_k(
            target_points, len(leukocytes), triangles, areas, radii, num_simulations, rng
        )
        k = {
            "k_um2": observed,
            "random_median_k_um2": np.median(simulated, axis=0),
            "random_lower_k_um2": np.percentile(simulated, 2.5, axis=0),
            "random_upper_k_um2": np.percentile(simulated, 97.5, axis=0),
        }
        k["k_ratio"] = k["k_um2"] / k["random_median_k_um2"]

        curves.append(
            pd.DataFrame(
                {
                    "target": target,
                    "distance_um": distances_um,
                    **{key: values[:-1] for key, values in k.items()},
                }
            )
        )
        summaries.append(
            {
                "target": target,
                "tissue": tissue or "all",
                "leukocytes": len(leukocytes),
                "targets": len(target_points),
                "area_um2": area_um2,
                "distance_um": summary_distance_um,
                **{key: values[-1] for key, values in k.items()},
            }
        )
        print(
            f"{target}: {len(target_points)} cells, "
            f"K ratio at {summary_distance_um:g} um = {k['k_ratio'][-1]:.2f}"
        )

    name = f"run_{run_id}_{tissue}" if tissue else f"run_{run_id}"
    pd.concat(curves).to_csv(save_dir / f"{name}_curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(save_dir / f"{name}_summary.csv", index=False)
    print(f"Saved colocalisation curves and summary to {save_dir}")


if __name__ == "__main__":
    typer.run(main)
