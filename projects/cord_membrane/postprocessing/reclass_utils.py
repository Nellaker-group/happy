import numpy as np
import torch

import happy.db.eval_runs_interface as db
from happy.cell_infer.cell_infer import setup_model
from happy.utils.utils import get_device
from projects.cord_membrane.postprocessing.reclassed_file import load_hdf5_data


def reclass_cells(
    cell_predictions,
    cell_embeddings,
    tissue_predictions,
    organ,
    valid_cells_by_tissue,
    cell_model_id,
    batch_size=100000,
):
    """Reassign cells whose predicted type cannot occur in the tissue they sit in.

    Each invalid cell takes the highest scoring cell type that is allowed in its
    tissue, using the output layer of the cell classifier on the stored embeddings.
    Cells in tissues missing from valid_cells_by_tissue (e.g. Unlabelled) are kept.

    Args:
        cell_predictions: (N,) cell class ids from cell inference
        cell_embeddings: (N, 64) raw (not standardised) cell embeddings
        tissue_predictions: (N,) tissue class ids
        organ: organ defining the cell and tissue ids
        valid_cells_by_tissue: tissue label -> list of allowed cell labels
        cell_model_id: id of the cell model that produced the embeddings
        batch_size: number of embeddings passed through the output layer at once
    Returns:
        (N,) array of reassigned cell class ids
    """
    cell_predictions = np.asarray(cell_predictions).astype(np.int64)
    tissue_predictions = np.asarray(tissue_predictions).astype(np.int64)

    allowed = _allowed_cells_by_tissue(organ, valid_cells_by_tissue)
    scores = _cell_scores(cell_embeddings, cell_model_id, len(organ.cells), batch_size)
    _check_scores_match_predictions(scores, cell_predictions)

    cell_allowed = allowed[tissue_predictions]
    invalid = ~cell_allowed[np.arange(len(cell_predictions)), cell_predictions]
    masked_scores = np.where(cell_allowed[invalid], scores[invalid], -np.inf)

    updated_predictions = cell_predictions.copy()
    updated_predictions[invalid] = masked_scores.argmax(axis=1)
    print(
        f"Reassigned {invalid.sum()} of {len(cell_predictions)} cells "
        f"({invalid.mean() * 100:.2f}%) to a cell type valid for their tissue"
    )
    return updated_predictions


def get_raw_cell_embeddings(
    project_name, run_id, coords, x_min, y_min, width, height, custom_path=None
):
    """Load raw cell embeddings from the cell inference HDF5 file.

    The tissue HDF5 file stores standardised embeddings when tissue inference
    was run with standardisation, so the raw ones are read from the cell file.
    Both files are sorted by coordinates, so rows match once the same patch is used.
    """
    cell_data = load_hdf5_data(
        project_name,
        run_id,
        x_min,
        y_min,
        width,
        height,
        tissue=False,
        custom_path=custom_path,
    )
    if not np.array_equal(cell_data.coords, coords):
        raise ValueError(
            "Cell and tissue HDF5 files do not contain the same cells in the same order"
        )
    return cell_data.cell_embeddings


def get_cell_model_id(run_id):
    cell_model_id = db.get_eval_run_by_id(run_id).cell_model_id
    if cell_model_id is None:
        raise ValueError(f"No cell model recorded for run {run_id}")
    return cell_model_id


def _allowed_cells_by_tissue(organ, valid_cells_by_tissue):
    # (num_tissues, num_cells) mask of cell types allowed in each tissue
    tissue_ids = {tissue.label: tissue.id for tissue in organ.tissues}
    cell_ids = {cell.label: cell.id for cell in organ.cells}
    allowed = np.ones((len(organ.tissues), len(organ.cells)), dtype=bool)
    for tissue_label, cell_labels in valid_cells_by_tissue.items():
        allowed[tissue_ids[tissue_label]] = False
        allowed[tissue_ids[tissue_label], [cell_ids[c] for c in cell_labels]] = True
    return allowed


def _cell_scores(cell_embeddings, cell_model_id, num_cells, batch_size):
    device = get_device()
    model, _ = setup_model(cell_model_id, num_cells, device)
    # resnet heads are stored under fc, inceptionresnetv2 under last_linear
    head = model.fc if hasattr(model, "fc") else model.last_linear
    output_layer = head.output_layer.to(device).eval()

    scores = []
    with torch.no_grad():
        for start in range(0, len(cell_embeddings), batch_size):
            batch = torch.as_tensor(
                cell_embeddings[start : start + batch_size], dtype=torch.float32
            ).to(device)
            scores.append(output_layer(batch).cpu().numpy())
    return np.concatenate(scores)


def _check_scores_match_predictions(scores, cell_predictions, min_agreement=0.99):
    # the output layer should reproduce the stored predictions from the embeddings
    agreement = np.mean(scores.argmax(axis=1) == cell_predictions)
    if agreement < min_agreement:
        raise ValueError(
            f"Output layer reproduces only {agreement * 100:.1f}% of stored cell "
            f"predictions. Check the cell model id and that embeddings are not standardised."
        )
