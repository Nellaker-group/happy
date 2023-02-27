import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from happy.utils.utils import load_image
from happy.data.utils import group_annotations_by_image


class NucleiDataset(Dataset):
    def __init__(
        self,
        annotations_dir,
        dataset_names,
        split="train",
        transform=None,
    ):
        """
        Args:
            annotations_dir (Path): path to directory with all annotation files
            dataset_names (list): list of directory names of datasets
            split (string): One of "train", "val", "test", "all"
            transform: transforms to apply to the data
        """
        self.annotations_dir = annotations_dir
        self.dataset_names = dataset_names
        self.split = split
        self.transform = transform

        self.dataset_names = self._load_datasets(dataset_names)
        self.classes = self._load_classes()
        self.ungrouped_annotations = self._load_annotations()

        self.all_annotations = group_annotations_by_image(self.ungrouped_annotations)

    def __len__(self):
        return len(self.all_annotations)

    def __getitem__(self, idx):
        img = load_image(self.all_annotations["image_path"][idx])
        annot = self.get_annotations_in_image(idx)
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
        return {"nucleus": 0}

    def get_annotations_in_image(self, image_index):
        # get ground truth annotations
        image_data = self.all_annotations.iloc[image_index]

        # for images without annotations
        if np.isnan(image_data["x1"][0]):
            return np.zeros((0, 5))

        # extract all annotations in the image
        x1s = np.array(image_data["x1"])
        y1s = np.array(image_data["y1"])
        x2s = np.array(image_data["x2"])
        y2s = np.array(image_data["y2"])
        class_names = np.array(image_data["class_name"])
        class_names = np.vectorize(self.classes.get)(class_names)
        return np.column_stack((x1s, y1s, x2s, y2s, class_names))

    def _load_annotations(self):
        df_list = []
        for dataset_name in self.dataset_names:
            file_path = self.annotations_dir / dataset_name / f"{self.split}_nuclei.csv"

            annotations = pd.read_csv(
                file_path, names=["image_path", "x1", "y1", "x2", "y2", "class_name"]
            )
            assert np.where(annotations["x1"] > annotations["x2"])[0].size == 0
            assert np.where(annotations["y1"] > annotations["y2"])[0].size == 0
            df_list.append(annotations)
        return pd.concat(df_list, ignore_index=True)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.all_annotations["image_path"][image_index])
        return float(image.width) / float(image.height)
