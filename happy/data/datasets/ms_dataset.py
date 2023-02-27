from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import IterableDataset

from happy.utils.utils import process_image


class MSDataset(IterableDataset, ABC):
    def __init__(self, microscopefile, remaining_data, transform=None):
        self.file = microscopefile
        self.remaining_data = remaining_data
        self.transform = transform

        self.target_width = self.file.target_tile_width
        self.target_height = self.file.target_tile_height
        self.rescale_ratio = self.file.rescale_ratio
        self.start = 0
        self.end = len(self.remaining_data)

    # called by a dataloader. Uses torch workers to load data onto a gpu
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self._iter_data(self.start, self.end)
        else:
            # splits the datasets each worker gets proportional to the number of workers
            per_worker = int(
                np.math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

            return self._iter_data(iter_start, iter_end)

    @abstractmethod
    def _iter_data(self, iter_start, iter_end):
        pass

    @abstractmethod
    def _get_dataset_section(self, target_w, target_h, tile_range):
        pass


class NucleiDataset(MSDataset):
    def _iter_data(self, iter_start, iter_end):
        for img, tile_index, empty_tile in self._get_dataset_section(
            target_w=self.target_width,
            target_h=self.target_height,
            tile_range=(iter_start, iter_end),
        ):
            if not empty_tile:
                img = process_image(img).astype(np.float32) / 255.0
            sample = {
                "img": img,
                "tile_index": tile_index,
                "empty_tile": empty_tile,
                "scale": None,
                "annot": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            }
            if self.transform and not empty_tile:
                sample = self.transform(sample)
            yield sample

    # Generator to create a datasets of tiles within a range
    def _get_dataset_section(self, target_w, target_h, tile_range):
        tile_coords = self.remaining_data[tile_range[0] : tile_range[1]]
        for _dict in tile_coords:
            img = self.file.get_tile_by_coords(
                _dict["tile_x"], _dict["tile_y"], target_w, target_h
            )
            if self.file._img_is_empty(img):
                yield None, _dict["tile_index"], True
            else:
                yield img, _dict["tile_index"], False


class CellDataset(MSDataset):
    def _iter_data(self, iter_start, iter_end):
        for img, coord in self._get_dataset_section(
            target_w=200, target_h=200, tile_range=(iter_start, iter_end)
        ):
            img = process_image(img).astype(np.float32) / 255.0
            sample = {
                "img": img,
                "coord": coord,
                "annot": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            }
            if self.transform:
                sample = self.transform(sample)
            yield sample

    # Generator to create a datasets of tiles with cell centres within a range
    def _get_dataset_section(self, target_w, target_h, tile_range):
        # Returns 200x200 images by default for cell classifier centered on nuclei
        cell_coords = self.remaining_data[tile_range[0] : tile_range[1]]
        for _dict in cell_coords:
            img = self.file.get_cell_tile_by_cell_coords(
                _dict["x"], _dict["y"], target_w, target_h
            )
            yield img, (_dict["x"], _dict["y"])
