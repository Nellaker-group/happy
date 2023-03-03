from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from happy.utils.utils import load_image


class CellDataset(Dataset):
    def __init__(
        self,
        organ,
        annotations_dir,
        dataset_names,
        split="train",
        transform=None,
    ):
        """
        Args:
            organ (Organ): the organ to access the cell data from
            annotations_dir (Path): path to directory with all annotation files
            dataset_names (list): list of directory names of datasets
            split (string): One of "train", "val", "test", "all"
            transform: transforms to apply to the data
        """
        self.organ = organ
        self.annotations_dir = annotations_dir
        self.split = split
        self.transform = transform

        self.dataset_names = self._load_datasets(dataset_names)
        self.classes = self._load_classes()
        self.all_annotations = self._load_annotations()

        self.class_sampling_weights = self._get_class_sampling_weights()

    def __len__(self):
        return len(self.all_annotations)

    def __getitem__(self, idx):
        img = load_image(self.all_annotations["image_path"][idx])
        annot = self._get_class_in_image(idx)
        sample = {"img": img, "annot": annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _load_datasets(self, dataset_names):
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names

    def _load_classes(self):
        return {cell.label: cell.id for cell in self.organ.cells}

    def _load_annotations(self):
        df_list = []
        for dataset_name in self.dataset_names:
            # Get the file path and oversampled file if specified
            dir_path = self.annotations_dir / dataset_name
            file_name = f"{self.split}_cell.csv"
            file_path = dir_path / file_name

            annotations = pd.read_csv(file_path, names=["image_path", "class_name"])
            assert annotations.class_name.isin(self.classes.keys()).all()
            df_list.append(annotations)
        return pd.concat(df_list, ignore_index=True)

    def _get_class_in_image(self, image_index):
        return self.classes[self.all_annotations["class_name"][image_index]]

    def _get_class_sampling_weights(self):
        cell_classes = self.all_annotations["class_name"]
        class_counts = defaultdict(int)
        for img_class in cell_classes:
            class_counts[img_class] += 1
        list_of_weights = [1 / class_counts[x] for x in cell_classes]
        return list_of_weights

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.all_annotations["image_path"][image_index])
        return float(image.width) / float(image.height)
