from typing import Optional

import typer

import happy.db.eval_runs_interface as db
from happy.hdf5 import HDF5Dataset
from happy.organs import get_organ
from happy.utils.utils import get_project_dir
from happy.graph.enums import FeatureArg, MethodArg
from happy.graph.graph_creation.create_graph import setup_graph
from projects.cord_membrane.graph_utils import remove_long_edges
from projects.cord_membrane.postprocessing.cord.reclass_tissue import REMNANTS, VESSELS
from projects.cord_membrane.postprocessing.reclassed_file import load_predictions
from projects.cord_membrane.spatial_analysis.csa.calculate_csa import (
    get_sections,
    get_structures,
    measure_areas,
    measure_vessels,
)
from projects.cord_membrane.spatial_analysis.csa.save import (
    save_areas,
    save_qupath_outlines,
    save_sections,
    save_vessels,
)


def main(
    run_id: int = typer.Option(...),
    project_name: str = "cord_membrane",
    db_name: str = "main.db",
    custom_path: Optional[str] = None,
    reclassed: bool = True,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 8,
    graph_method: MethodArg = MethodArg.intersection,
    max_section_edge_um: float = 230.0,
    max_vessel_edge_um: float = 140.0,
    min_section_size: int = 1000,
    min_structure_size: int = 100,
    min_endothelial_cells: int = 5,
    alpha_radius_um: float = 50.0,
    section_alpha_radius_um: float = 115.0,
    vitelline: bool = False,
):
    """Measure umbilical cord sections, their structures and their vessels.

    Saves the area, perimeter, diameters and eccentricity of each cord section,
    areas of each vessel (and remnant) and Wharton's jelly, the lumen, wall area,
    Wagenworth index and wall thickness of each vessel, and all outlines as
    QuPath GeoJSON annotations.

    Run after postprocessing.run_postprocessing (or after tissue inference with
    --no-reclassed):
        python -m projects.cord_membrane.spatial_analysis.csa.run_csa --run-id 1

    Args:
        run_id: eval run with cell and tissue predictions
        project_name: project dir containing the embeddings and results
        db_name: database containing the eval run
        custom_path: project dir to read embeddings from, if not project_name
        reclassed: use the reclassified predictions from run_postprocessing,
            otherwise the original predictions
        x_min: bottom left x coordinate of the region to measure (0 for full WSI)
        y_min: bottom left y coordinate of the region to measure (0 for full WSI)
        width: width of the region to measure (-1 for full WSI)
        height: height of the region to measure (-1 for full WSI)
        k: value of k for kNN graph edge construction
        graph_method: which type of edge construction to use
        max_section_edge_um: longest edge kept when finding cord sections
        max_vessel_edge_um: longest edge kept when finding vessels and remnants
        min_section_size: minimum number of cells for a cord section to be measured
        min_structure_size: minimum number of cells for a structure to be measured
        min_endothelial_cells: minimum endothelial cells for the collapsed lumen
            ellipse and wall thickness
        alpha_radius_um: alpha shape radius for vessel and remnant outlines; smaller
            follows edges tighter, lumens narrower than about twice this are
            treated as collapsed
        section_alpha_radius_um: alpha shape radius for cord section outlines,
            measured on their largest part
        vitelline: also measure allantoic and vitelline remnants separately from
            Wharton's jelly
    """
    db.init(db_name)
    organ = get_organ("cord")
    pixel_size = db.get_slide_pixel_size_by_evalrun(run_id)
    save_dir = get_project_dir(project_name) / "results" / "csa"
    save_dir.mkdir(parents=True, exist_ok=True)

    coords, cell_predictions, tissue_predictions = load_predictions(
        project_name,
        run_id,
        "cord",
        reclassed,
        x_min,
        y_min,
        width,
        height,
        custom_path,
    )
    graph = setup_graph(
        HDF5Dataset(coords=coords, cell_predictions=cell_predictions),
        organ,
        FeatureArg.predictions,
        k,
        graph_method,
    )
    edges = graph.edge_index.numpy()
    section_edges = remove_long_edges(edges, coords, max_section_edge_um / pixel_size)
    structure_edges = remove_long_edges(edges, coords, max_vessel_edge_um / pixel_size)

    sections = get_sections(section_edges, len(coords), min_section_size)
    print(f"Measuring {len(sections)} cord section(s)")
    tissue_labels = VESSELS + REMNANTS if vitelline else VESSELS
    structures = get_structures(
        structure_edges, tissue_predictions, organ, tissue_labels, min_structure_size
    )
    measurements = measure_areas(
        coords,
        sections,
        structures,
        alpha_radius_um / pixel_size,
        section_alpha_radius_um / pixel_size,
        pixel_size,
    )
    vessels = measure_vessels(
        measurements,
        coords,
        cell_predictions,
        organ,
        pixel_size,
        min_endothelial_cells,
    )

    save_sections(save_dir / f"run_{run_id}_sections.csv", measurements)
    save_areas(save_dir / f"run_{run_id}_areas.csv", measurements)
    save_vessels(save_dir / f"run_{run_id}_vessels.csv", vessels)
    save_qupath_outlines(
        save_dir / f"run_{run_id}_outlines.geojson", measurements, vessels, organ
    )


if __name__ == "__main__":
    typer.run(main)
