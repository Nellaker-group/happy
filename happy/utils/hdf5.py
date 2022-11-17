import numpy as np
import h5py

import happy.db.eval_runs_interface as db


def get_embeddings_file(project_dir, run_id):
    db.init()
    embeddings_dir = project_dir / "results" / "embeddings"
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    return embeddings_dir / embeddings_path


def get_hdf5_datasets(file_path, start, num_points, verbose=True):
    with h5py.File(file_path, "r") as f:
        subset_start = (
            int(len(f["predictions"]) * start) if 1 > start > 0 else int(start)
        )
        subset_end = (
            len(f["predictions"]) if num_points == -1 else subset_start + num_points
        )
        if verbose:
            print(f"Getting {subset_end - subset_start} datapoints from hdf5")
        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]
        coords = f["coords"][subset_start:subset_end]
        confidence = f["confidence"][subset_start:subset_end]
        return predictions, embeddings, coords, confidence, subset_start, subset_end


def get_datasets_in_patch(file_path, x_min, y_min, width, height, verbose=True):
    predictions, embeddings, coords, confidence, _, _ = get_hdf5_datasets(
        file_path, 0, -1, verbose=verbose
    )

    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        return predictions, embeddings, coords, confidence

    mask = np.logical_and(
        (np.logical_and(coords[:, 0] > x_min, (coords[:, 1] > y_min))),
        (
            np.logical_and(
                coords[:, 0] < (x_min + width), (coords[:, 1] < (y_min + height))
            )
        ),
    )

    patch_coords = coords[mask]
    patch_predictions = predictions[mask]
    patch_embeddings = embeddings[mask]
    patch_confidence = confidence[mask]

    return patch_predictions, patch_embeddings, patch_coords, patch_confidence


def filter_by_cell_type(predictions, embeddings, coords, confidence, cell_type, organ):
    label_map = {cell.label: cell.id for cell in organ.cells}
    filtered_embeddings = embeddings[predictions == label_map[cell_type]]
    filtered_predictions = predictions[predictions == label_map[cell_type]]
    filtered_confidence = confidence[predictions == label_map[cell_type]]
    filtered_coords = coords[predictions == label_map[cell_type]]

    return (
        filtered_predictions,
        filtered_embeddings,
        filtered_coords,
        filtered_confidence,
    )