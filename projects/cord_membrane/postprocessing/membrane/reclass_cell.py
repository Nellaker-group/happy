from happy.organs import get_organ
from projects.cord_membrane.postprocessing.reclass_utils import reclass_cells

# Cell types that can occur in each membrane tissue. Tissues not listed are left unchanged.
VALID_CELLS_BY_TISSUE = {
    "Amnion": ["EPI", "MES", "FIB", "WBC"],
    "CMembrane": ["MES", "FIB", "CYT", "EVT", "WBC"],
    "Decidua": ["VEN", "MES", "FIB", "VMY", "EVT", "MAT", "WBC"],
    "RVilli": ["MES", "FIB", "CYT", "EVT", "WBC"],
    "SArtery": ["VEN", "MES", "FIB", "VMY", "EVT", "MAT", "WBC"],
}


def reclass_cell(cell_predictions, cell_embeddings, tissue_predictions, cell_model_id):
    """Reassign membrane cells whose predicted type cannot occur in their tissue.

    Args:
        cell_predictions: (N,) cell class ids from cell inference
        cell_embeddings: (N, 64) raw cell embeddings (see get_raw_cell_embeddings)
        tissue_predictions: (N,) tissue class ids
        cell_model_id: id of the cell model that produced the embeddings
    Returns:
        (N,) array of reassigned cell class ids
    """
    return reclass_cells(
        cell_predictions,
        cell_embeddings,
        tissue_predictions,
        get_organ("membrane"),
        VALID_CELLS_BY_TISSUE,
        cell_model_id,
    )
