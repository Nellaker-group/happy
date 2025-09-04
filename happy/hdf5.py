from pathlib import Path
import os

import numpy as np
import numpy.typing as npt
import h5py

import happy.db.eval_runs_interface as db


class HDF5Dataset:
    def __init__(
        self,
        cell_predictions=None,
        cell_embeddings=None,
        cell_confidence=None,
        coords=None,
        tissue_predictions=None,
        tissue_embeddings=None,
        tissue_confidence=None,
        edge_index=None,
        start=None,
        end=None,
    ):
        self.cell_predictions = cell_predictions
        self.cell_embeddings = cell_embeddings
        self.cell_confidence = cell_confidence
        self.coords = coords
        self.tissue_predictions = tissue_predictions
        self.tissue_embeddings = tissue_embeddings
        self.tissue_confidence = tissue_confidence
        self.edge_index = None
        self.start = start
        self.end = end

    @staticmethod
    def from_path(file_path, start=0, num_points=-1) -> "HDF5Dataset":
        dataset = HDF5Dataset()
        dataset._load_hdf5_datasets(file_path, start, num_points)
        return dataset

    def to_path(self, file_path):
        if not os.path.isfile(file_path):
            total = len(self.cell_predictions)
            with h5py.File(file_path, "w-") as f:
                f.create_dataset(
                    "cell_predictions",
                    data=self.cell_predictions,
                    shape=(total,),
                    dtype="int8",
                )
                f.create_dataset(
                    "cell_embeddings",
                    data=self.cell_embeddings,
                    shape=(total, 64),
                    dtype="float32",
                )
                f.create_dataset(
                    "cell_confidence",
                    data=self.cell_confidence,
                    shape=(total,),
                    dtype="float16",
                )
                f.create_dataset(
                    "coords", data=self.coords, shape=(total, 2), dtype="uint32"
                )
                f.create_dataset(
                    "tissue_predictions",
                    data=self.tissue_predictions,
                    shape=(total,),
                    dtype="int8",
                )
                f.create_dataset(
                    "tissue_embeddings",
                    data=self.tissue_embeddings,
                    shape=(total, self.tissue_embeddings.shape[1]),
                    dtype="float32",
                )
                f.create_dataset(
                    "tissue_confidence",
                    data=self.tissue_confidence,
                    shape=(total,),
                    dtype="float16",
                )
                f.create_dataset(
                    "edge_index",
                    data=self.edge_index,
                    shape=(2, self.edge_index.shape[1]),
                    dtype="int32",
                )
        else:
            print(f"File at {file_path} already exists, skipping saving")

    def _apply_mask(self, mask) -> "HDF5Dataset":
        self.cell_embeddings = self.cell_embeddings[mask]
        self.cell_predictions = self.cell_predictions[mask]
        self.cell_confidence = self.cell_confidence[mask]
        self.coords = self.coords[mask]
        if self.tissue_predictions is not None:
            self.tissue_predictions = self.tissue_predictions[mask]
            self.tissue_embeddings = self.tissue_embeddings[mask]
            self.tissue_confidence = self.tissue_confidence[mask]
        return self

    def filter_by_patch(self, x_min, y_min, width, height) -> "HDF5Dataset":
        if x_min == 0 and y_min == 0 and width == -1 and height == -1:
            return self
        mask = np.logical_and(
            (np.logical_and(self.coords[:, 0] > x_min, (self.coords[:, 1] > y_min))),
            (
                np.logical_and(
                    self.coords[:, 0] < (x_min + width),
                    (self.coords[:, 1] < (y_min + height)),
                )
            ),
        )
        return self._apply_mask(mask)

    def filter_by_confidence(self, min_conf, max_conf) -> ("HDF5Dataset", npt.ArrayLike):
        mask = np.logical_and(
            (self.cell_confidence >= min_conf), (self.cell_confidence <= max_conf)
        )
        return self._apply_mask(mask), mask

    def filter_by_cell_type(self, cell_type, organ) -> ("HDF5Dataset", npt.ArrayLike):
        label_map = {cell.label: cell.id for cell in organ.cells}
        mask = self.cell_predictions == label_map[cell_type]
        return self._apply_mask(mask), mask

    def filter_randomly(self, percent_to_remove) -> ("HDF5Dataset", npt.ArrayLike):
        num_to_remove = int(len(self.cell_predictions) * percent_to_remove)
        mask = np.ones(len(self.cell_predictions), dtype=bool)
        remove_indices = np.random.choice(
            np.arange(len(self.cell_predictions)), num_to_remove, replace=False
        )
        mask[remove_indices] = False
        return self._apply_mask(mask), mask

    def standardise_cell_features(self):
        mean = np.mean(self.cell_embeddings, axis=0, keepdims=True)
        std = np.std(self.cell_embeddings, axis=0, keepdims=True)
        self.cell_embeddings = (self.cell_embeddings - mean) / std
        return self

    def sort_by_coordinates(self) -> "HDF5Dataset":
        sort_args = np.lexsort((self.coords[:, 1], self.coords[:, 0]))
        print("Data sorted by x coordinates")
        return self._apply_mask(sort_args)

    def _load_hdf5_datasets(self, file_path, start, num_points):
        with h5py.File(file_path, "r") as f:
            subset_start = (
                int(len(f["coords"]) * start) if 1 > start > 0 else int(start)
            )
            subset_end = (
                len(f["coords"]) if num_points == -1 else subset_start + num_points
            )
            self.start = subset_start
            self.end = subset_end
            self.coords = f["coords"][subset_start:subset_end]
            if "tissue" not in file_path.name:
                self.cell_predictions = f["predictions"][subset_start:subset_end]
                self.cell_embeddings = f["embeddings"][subset_start:subset_end]
                self.cell_confidence = f["confidence"][subset_start:subset_end]
            elif "tissue" in file_path.name:
                self.cell_predictions = f["cell_predictions"][subset_start:subset_end]
                self.cell_embeddings = f["cell_embeddings"][subset_start:subset_end]
                self.cell_confidence = f["cell_confidence"][subset_start:subset_end]
                self.tissue_predictions = f["tissue_predictions"][
                    subset_start:subset_end
                ]
                self.tissue_embeddings = f["tissue_embeddings"][subset_start:subset_end]
                self.tissue_confidence = f["tissue_confidence"][subset_start:subset_end]


def get_embeddings_file(project_name, run_id, tissue=False, custom_path=None):
    if custom_path:
        # custom path to project directory, use directly
        embeddings_dir = Path(custom_path) / "results" / "embeddings"
    else:
        embeddings_dir = (
            Path(__file__).parent.parent
            / "projects" 
            / project_name
            / "results"
            / "embeddings"
        )

    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    if tissue:
        base_name = embeddings_path.stem
        file_name = f"{base_name}_tissues.hdf5"
        embeddings_path = embeddings_path.parent / file_name
    
    return embeddings_dir / embeddings_path