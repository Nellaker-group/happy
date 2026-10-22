from pathlib import Path

import h5py

import happy.db.eval_runs_interface as db
from happy.hdf5 import HDF5Dataset, get_embeddings_file
from happy.utils.utils import get_project_dir

# the rewritten hdf5 file saves the preedictions and coords etc, but not the embeddings to save space
# so if needing embeddings, take original


def get_reclassed_path(project_name, run_id):
    """Path of the reclassified predictions file in project_name's embeddings dir.

    """
    cell_path = get_embeddings_file(project_name, run_id)
    return cell_path.with_name(f"{cell_path.stem}_reclassed.hdf5")


def find_embeddings_file(project_name, run_id, tissue=False, custom_path=None):
    """A run's cell (or tissue) HDF5 file, from custom_path if it is there, otherwise project_name."""
    eval_run = db.get_eval_run_by_id(run_id)
    if not eval_run.embeddings_path:
        raise ValueError(f"No embeddings path recorded for run {run_id}")
    cell_path = Path(eval_run.embeddings_path)
    names = [cell_path]
    if tissue:
        names = [cell_path.with_name(f"{cell_path.stem}_tissues.hdf5")]
        if eval_run.tissue_embeddings_path:
            names.insert(0, Path(eval_run.tissue_embeddings_path))

    roots = [Path(custom_path)] if custom_path else []
    roots.append(get_project_dir(project_name))
    candidates = [root / "results" / "embeddings" / name for root in roots for name in names]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in candidates]}")


def load_hdf5_data(
    project_name, run_id, x_min, y_min, width, height, tissue=False, custom_path=None
):
    """A run's HDF5 data (see find_embeddings_file), filtered to a patch and sorted by coordinates."""
    path = find_embeddings_file(project_name, run_id, tissue, custom_path)
    print(f"Getting data from: {path}")
    hdf5_data = HDF5Dataset.from_path(path)
    return hdf5_data.filter_by_patch(x_min, y_min, width, height).sort_by_coordinates()


def save_reclassed(save_path, coords, cell_predictions, tissue_predictions, settings):
    """Save reclassified predictions, with the settings used as file attributes."""
    if save_path.exists():
        print(f"Overwriting existing file at {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path, "w") as f:
        f.create_dataset("coords", data=coords, dtype="uint32")
        f.create_dataset("cell_predictions", data=cell_predictions, dtype="int8")
        f.create_dataset("tissue_predictions", data=tissue_predictions, dtype="int8")
        for key, value in settings.items():
            f.attrs[key] = "" if value is None else value
    print(f"Saved reclassified predictions to {save_path}")


def load_predictions(
    project_name,
    run_id,
    organ_name,
    reclassed,
    x_min,
    y_min,
    width,
    height,
    custom_path=None,
):
    """Cell coordinates with their cell and tissue predictions, sorted by coordinates.

    Reads the reclassified predictions saved by run_postprocessing if reclassed,
    otherwise the original predictions from the tissue HDF5 file.

    Returns:
        coords, cell_predictions, tissue_predictions
    """
    if not reclassed:
        hdf5_data = load_hdf5_data(
            project_name,
            run_id,
            x_min,
            y_min,
            width,
            height,
            tissue=True,
            custom_path=custom_path,
        )
        return hdf5_data.coords, hdf5_data.cell_predictions, hdf5_data.tissue_predictions

    path = get_reclassed_path(project_name, run_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No reclassified predictions at {path}. Run "
            f"postprocessing.run_postprocessing first, or use --no-reclassed"
        )
    print(f"Getting reclassified predictions from: {path}")
    with h5py.File(path, "r") as f:
        if f.attrs["organ_name"] != organ_name:
            raise ValueError(
                f"{path} was reclassified for {f.attrs['organ_name']}, not {organ_name}"
            )
        coords = f["coords"][:]
        cell_predictions = f["cell_predictions"][:]
        tissue_predictions = f["tissue_predictions"][:]

    # same patch filter as HDF5Dataset.filter_by_patch
    if not (x_min == 0 and y_min == 0 and width == -1 and height == -1):
        mask = (
            (coords[:, 0] > x_min)
            & (coords[:, 1] > y_min)
            & (coords[:, 0] < x_min + width)
            & (coords[:, 1] < y_min + height)
        )
        coords = coords[mask]
        cell_predictions = cell_predictions[mask]
        tissue_predictions = tissue_predictions[mask]
    return coords, cell_predictions, tissue_predictions
