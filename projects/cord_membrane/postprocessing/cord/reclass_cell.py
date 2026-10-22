from happy.organs import get_organ
from projects.cord_membrane.postprocessing.reclass_utils import reclass_cells

# Cell types that can occur in each cord tissue. Tissues not listed are left unchanged.
VALID_CELLS_BY_TISSUE = {
    "AmnEpi": ["EPI", "MES", "FIB", "MFB", "WBC"],
    "SWJelly": ["EPI", "MES", "FIB", "MFB", "WBC"],
    "IWJelly": ["MES", "FIB", "MFB", "VMY", "WBC"],
    "PWJelly": ["MES", "FIB", "MFB", "VMY", "WBC"],
    "UArtery": ["VEN", "MES", "FIB", "MFB", "VMY", "WBC"],
    "UVein": ["VEN", "MES", "FIB", "MFB", "VMY", "WBC"],
    "Allantois": ["EPI", "MES", "FIB", "MFB", "WBC"],
    "VitVes": ["EPI", "MES", "FIB", "MFB", "VMY", "WBC"],
    "VitDuct": ["EPI", "MES", "FIB", "MFB", "WBC"],
}


def reclass_cell(cell_predictions, cell_embeddings, tissue_predictions, cell_model_id):
    """Reassign cord cells whose predicted type cannot occur in their tissue.

    Args:
        cell_predictions: (N,) cell class ids from cell inference
        cell_embeddings: (N, 64) raw cell embeddings (see get_raw_cell_embeddings)
        tissue_predictions: (N,) tissue class ids, after reclass_tissue if used
        cell_model_id: id of the cell model that produced the embeddings
    Returns:
        (N,) array of reassigned cell class ids
    """
    return reclass_cells(
        cell_predictions,
        cell_embeddings,
        tissue_predictions,
        get_organ("cord"),
        VALID_CELLS_BY_TISSUE,
        cell_model_id,
    )
