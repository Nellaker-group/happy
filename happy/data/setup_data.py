import albumentations as al
from torchvision import transforms

from happy.data.transforms.agumentations import AlbAugmenter, StainAugment
from happy.data.datasets.cell_dataset import CellDataset
from happy.data.datasets.nuclei_dataset import NucleiDataset
from happy.data.transforms.transforms import Normalizer, Resizer
from happy.data.transforms.utils.color_conversion import get_rgb_matrices


def setup_nuclei_datasets(annot_dir, dataset_names, multiple_val_sets, test_set=False):
    if not test_set:
        # Create the datasets from all directories specified in dataset_names
        dataset_train = get_nuclei_dataset("train", annot_dir, dataset_names)
        dataset_val = get_nuclei_dataset("val", annot_dir, dataset_names)
        datasets = {"train": dataset_train, "val_all": dataset_val}
        # Create validation datasets from all directories specified in dataset_names
        dataset_val_dict = {}
        if multiple_val_sets:
            for dataset_name in dataset_names:
                dataset_val_dict[dataset_name] = get_nuclei_dataset(
                    "val", annot_dir, dataset_name
                )
            datasets.update(dataset_val_dict)
    else:
        dataset_test = get_nuclei_dataset("test", annot_dir, dataset_names)
        datasets = {"test": dataset_test}
    print("Dataset configured")
    return datasets


def setup_cell_datasets(
    organ,
    annot_dir,
    dataset_names,
    image_size,
    multiple_val_sets,
    test_set=False,
):
    if not test_set:
        # Create the datasets from all directories specified in dataset_names
        dataset_train = get_cell_dataset(
            organ, "train", annot_dir, dataset_names, image_size
        )
        dataset_val = get_cell_dataset(
            organ, "val", annot_dir, dataset_names, image_size
        )
        datasets = {"train": dataset_train, "val_all": dataset_val}
        # Create validation datasets from all directories specified in dataset_names
        dataset_val_dict = {}
        if multiple_val_sets:
            for dataset_name in dataset_names:
                dataset_val_dict[dataset_name] = get_cell_dataset(
                    organ, "val", annot_dir, dataset_name, image_size
                )
            datasets.update(dataset_val_dict)
    else:
        dataset_test = get_cell_dataset(
            organ, "test", annot_dir, dataset_names, image_size
        )
        datasets = {"test": dataset_test}
    print("Dataset configured")
    return datasets


def get_nuclei_dataset(split, annot_dir, dataset_names):
    augmentations = True if split == "train" else False
    transform = _setup_transforms(augmentations, nuclei=True)
    dataset = NucleiDataset(
        annotations_dir=annot_dir,
        dataset_names=dataset_names,
        split=split,
        transform=transform,
    )
    return dataset


def get_cell_dataset(organ, split, annot_dir, dataset_names, image_size):
    augmentations = True if split == "train" else False
    transform = _setup_transforms(augmentations, image_size=image_size, nuclei=False)
    dataset = CellDataset(
        organ=organ,
        annotations_dir=annot_dir,
        dataset_names=dataset_names,
        split=split,
        transform=transform,
    )
    return dataset


def _setup_transforms(augmentations, image_size=None, nuclei=True):
    transform = []
    if augmentations:
        transform.append(_augmentations(nuclei))
    transform.append(Normalizer())
    if nuclei:
        transform.append(Resizer())
    else:
        transform.append(
            Resizer(
                min_side=image_size[0],
                max_side=image_size[1],
                padding=False,
                scale_annotations=False,
            )
        )
    return transforms.Compose(transform)


def _augmentations(nuclei):
    alb = [
        al.Flip(p=0.5),
        al.RandomRotate90(p=0.5),
        StainAugment(get_rgb_matrices(), p=0.9, variance=0.4),
        al.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.8),
        al.GaussNoise(var_limit=(10.0, 200.0), p=0.8),
        al.Blur(blur_limit=5, p=0.8),
    ]
    return AlbAugmenter(list_of_albumentations=alb, bboxes=nuclei)
