from typing import Optional

import typer

import happy.db.eval_runs_interface as db
from happy.organs import get_organ
from happy.graph.enums import FeatureArg, MethodArg
from happy.graph.graph_creation.create_graph import setup_graph
from projects.cord_membrane.graph_utils import remove_long_edges
from projects.cord_membrane.postprocessing.cord.reclass_cell import (
    reclass_cell as reclass_cord_cell,
)
from projects.cord_membrane.postprocessing.cord.reclass_tissue import (
    reclass_remnants,
    reclass_tissue,
)
from projects.cord_membrane.postprocessing.membrane.reclass_cell import (
    reclass_cell as reclass_membrane_cell,
)
from projects.cord_membrane.postprocessing.reclass_utils import (
    get_cell_model_id,
    get_raw_cell_embeddings,
)
from projects.cord_membrane.postprocessing.reclassed_file import (
    get_reclassed_path,
    load_hdf5_data,
    save_reclassed,
)

RECLASS_CELL = {
    "placenta_cord": reclass_cord_cell,
    "placenta_membrane": reclass_membrane_cell,
}


def main(
    run_id: int = typer.Option(...),
    organ_name: str = typer.Option(..., help="placenta_cord or placenta_membrane"),
    project_name: str = "cord_membrane",
    db_name: str = "main.db",
    custom_path: Optional[str] = None,
    relabel_vessels: bool = True,
    relabel_remnants: bool = False,
    reclass_cells: bool = True,
    cell_model_id: Optional[int] = None,
    k: int = 8,
    graph_method: MethodArg = MethodArg.intersection,
    max_section_edge_um: float = 230.0,
    max_vessel_edge_um: float = 140.0,
    min_vessel_size: int = 500,
):
    """Reclassify a run's tissue and cell predictions into run_X_reclassed.hdf5.

    Cord: arteries and veins are relabelled with the 2:1 rule, smaller vessel
    fragments (and optionally remnants) take the tissue around them, then cells
    whose type cannot occur in their tissue are reassigned. Membrane: cells are
    reassigned. The original predictions in the database and HDF5 files are not
    changed. CSA and colocalisation read the reclassified file.

    Run after tissue inference:
        python -m projects.cord_membrane.postprocessing.run_postprocessing \\
            --run-id 1 --organ-name placenta_cord

    Args:
        run_id: eval run with cell and tissue predictions
        organ_name: placenta_cord or placenta_membrane (membrane rolls)
        project_name: project dir containing the embeddings
        db_name: database containing the eval run
        custom_path: project dir to read embeddings from, if not project_name
        relabel_vessels: cord only, relabel arteries and veins with the 2:1 rule
        relabel_remnants: cord only, relabel allantoic and vitelline remnants with
            the tissue around them
        reclass_cells: reassign cell types that cannot occur in their tissue
        cell_model_id: cell model of the run, looked up from the eval run if not given
        k: value of k for the graph used to relabel tissues
        graph_method: edge construction for the graph used to relabel tissues
        max_section_edge_um: longest edge kept when finding cord sections
        max_vessel_edge_um: longest edge kept when finding vessels and remnants
        min_vessel_size: minimum number of cells for a vessel to be relabelled
    """
    if organ_name not in RECLASS_CELL:
        raise typer.BadParameter(f"organ_name must be one of {list(RECLASS_CELL)}")
    db.init(db_name)
    organ = get_organ(organ_name)
    pixel_size = db.get_slide_pixel_size_by_evalrun(run_id)
    hdf5_data = load_hdf5_data(
        project_name, run_id, 0, 0, -1, -1, tissue=True, custom_path=custom_path
    )
    coords = hdf5_data.coords

    tissue_predictions = hdf5_data.tissue_predictions
    is_cord = organ_name == "placenta_cord"
    relabel_vessels = relabel_vessels and is_cord
    relabel_remnants = relabel_remnants and is_cord
    if relabel_vessels or relabel_remnants:
        graph = setup_graph(hdf5_data, organ, FeatureArg.predictions, k, graph_method)
        edges = graph.edge_index.numpy()
        vessel_edges = remove_long_edges(edges, coords, max_vessel_edge_um / pixel_size)
        if relabel_remnants:
            tissue_predictions = reclass_remnants(tissue_predictions, vessel_edges)
        if relabel_vessels:
            section_edges = remove_long_edges(
                edges, coords, max_section_edge_um / pixel_size
            )
            tissue_predictions = reclass_tissue(
                tissue_predictions, coords, section_edges, vessel_edges, min_vessel_size
            )

    cell_predictions = hdf5_data.cell_predictions
    if reclass_cells:
        cell_embeddings = get_raw_cell_embeddings(
            project_name, run_id, coords, 0, 0, -1, -1, custom_path
        )
        cell_model_id = cell_model_id or get_cell_model_id(run_id)
        cell_predictions = RECLASS_CELL[organ_name](
            cell_predictions, cell_embeddings, tissue_predictions, cell_model_id
        )

    settings = {
        "organ_name": organ_name,
        "relabel_vessels": relabel_vessels,
        "relabel_remnants": relabel_remnants,
        "reclass_cells": reclass_cells,
        "cell_model_id": cell_model_id if reclass_cells else None,
        "k": k,
        "graph_method": graph_method.value,
        "max_section_edge_um": max_section_edge_um,
        "max_vessel_edge_um": max_vessel_edge_um,
        "min_vessel_size": min_vessel_size,
    }
    save_reclassed(
        get_reclassed_path(project_name, run_id),
        coords,
        cell_predictions,
        tissue_predictions,
        settings,
    )


if __name__ == "__main__":
    typer.run(main)
