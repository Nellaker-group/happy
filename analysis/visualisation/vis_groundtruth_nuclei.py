from typing import Optional
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import typer

from happy.data.utils import draw_box, draw_centre, group_annotations_by_image
from happy.organs import get_organ


class ShapeArg(str, Enum):
    box = "box"
    point = "point"


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    annot_file: str = typer.Option(...),
    shape: ShapeArg = ShapeArg.point,
    num_images: Optional[int] = None,
    single_image_path: Optional[str] = None,
):
    """Visualises ground truth boxes or points from annotations for one image
    Args:
        project_name: name of the project dir to save visualisations to
        organ_name: name of organ
        dataset_name: the datasets to visualise predictions for
        annot_file: name of annotation file from which to get ground truth
        shape: one of 'box' or 'point' for visualising the prediction
        num_images: max number of images to generate. If None does all of them.
        single_image_path: if just running over on image. Path relative to project
    """
    organ = get_organ(organ_name)

    project_dir = Path(__file__).parent.parent.parent / "projects" / project_name
    dataset_dir = project_dir / "annotations" / "nuclei" / dataset_name
    annotation_path = dataset_dir / annot_file

    all_annotations = pd.read_csv(
        annotation_path, names=["image_path", "x1", "y1", "x2", "y2", "class_name"]
    )
    if single_image_path:
        single_image_annotations = all_annotations[
            all_annotations["image_path"] == single_image_path
        ].reset_index(drop=True)
        grouped_annotations = group_annotations_by_image(single_image_annotations)
    else:
        grouped_annotations = group_annotations_by_image(all_annotations)
        if num_images:
            grouped_annotations = grouped_annotations[:num_images]
    for index, row in grouped_annotations.iterrows():
        image_path = row["image_path"]
        img = cv2.imread(str(project_dir / image_path))
        x1s = row['x1']
        x2s = row['y1']
        y1s = row['x2']
        y2s = row['y2']
        labels = row['class_name']

        for i in range(len(x1s)):
            x1 = x1s[i]
            y1 = x2s[i]
            x2 = y1s[i]
            y2 = y2s[i]
            label_name = labels[i]

            if np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2):
                continue

            if shape.value == "point":
                draw_centre(img, x1, y1, x2, y2, label_name, organ, cell=False)
            elif shape.value == "box":
                draw_box(img, x1, y1, x2, y2, label_name, organ, cell=False)
            else:
                raise ValueError(f"No such draw shape {shape.value}")

        save_dir = (
            project_dir
            / "visualisations"
            / f"{annotation_path.parts[-3]}"
            / "groundtruth"
            / f"{annotation_path.parts[-2]}_gt"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        image_name = image_path.split("/")[-1]
        save_path = save_dir / image_name

        print(f"saving to: {save_path}")
        cv2.imwrite(str(save_path), img)


if __name__ == "__main__":
    typer.run(main)
