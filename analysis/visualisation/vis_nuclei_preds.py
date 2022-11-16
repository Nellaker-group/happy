from pathlib import Path
from enum import Enum
import os
from collections import namedtuple

import typer
import pandas as pd
import cv2
import torch

from happy.data.transforms.transforms import untransform_image
from happy.utils.utils import get_device
from happy.data.utils import draw_box, draw_centre
from happy.microscopefile.prediction_saver import PredictionSaver
from happy.train.nuc_train import setup_data, setup_model


class ShapeArg(str, Enum):
    box = "box"
    point = "point"


def main(
    project_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    shape: ShapeArg = ShapeArg.point,
    dataset_name: str = typer.Option(...),
    score_threshold: float = 0.4,
    max_detections: int = 500,
    num_images: int = 10,
    plot_groundtruth: bool = False,
):
    """Visualises network predictions as boxes or points for one datasets

    Args:
        project_name: name of the project dir to save visualisations to
        annot_dir: relative path to annotations
        pre_trained: relative path to pretrained model
        shape: one of 'box' or 'point' for visualising the prediction
        dataset_name: the datasets who's validation set to evaluate over
        score_threshold: the confidence threshold below which to discard predictions
        max_detections: number of maximum detections to save, ordered by score
        num_images: the number of images to evaluate
        plot_groundtruth: whether to plot ground truth points as well
    """
    device = get_device()

    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )
    os.chdir(str(project_dir))

    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(dataset_name, 1)

    dataloaders = setup_data(project_dir / annot_dir, hp, False, 3, 1)
    dataloaders.pop("train")
    model = setup_model(False, device, False, pre_trained)
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloaders["val_all"]):
            if idx >= num_images:
                break

            scale = data["scale"]

            scores, _, boxes = model(data["img"].to(device).float(), device)
            scores = scores.cpu().numpy()
            boxes = boxes.cpu().numpy()
            boxes /= scale

            filtered_preds = PredictionSaver.filter_by_score(
                max_detections, score_threshold, scores, boxes
            )
            gt_predictions = pd.DataFrame(
                data["annot"].numpy()[0][:, :4], columns=["x1", "y1", "x2", "y2"]
            )
            gt_predictions /= scale[0]

            img = untransform_image(data["img"][0])

            save_dir = (
                project_dir
                / "visualisations"
                / "nuclei"
                / "pred"
                / f"{dataset_name}_pred"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"val_{idx}.png"

            if plot_groundtruth:
                for i in range(len(gt_predictions.index)):
                    x1 = gt_predictions["x1"][i]
                    y1 = gt_predictions["y1"][i]
                    x2 = gt_predictions["x2"][i]
                    y2 = gt_predictions["y2"][i]
                    label = "nucleus"

                    if shape.value == "point":
                        draw_centre(
                            img, x1, y1, x2, y2, label, None, False, (0, 255, 255)
                        )
                    elif shape.value == "box":
                        draw_box(img, x1, y1, x2, y2, label, None, cell=False)
                    else:
                        raise ValueError(f"No such draw shape {shape.value}")

            if len(filtered_preds) != 0:
                all_predictions = pd.DataFrame(
                    filtered_preds, columns=["x1", "y1", "x2", "y2"]
                )

                for i in range(len(all_predictions.index)):
                    x1 = all_predictions["x1"][i]
                    y1 = all_predictions["y1"][i]
                    x2 = all_predictions["x2"][i]
                    y2 = all_predictions["y2"][i]
                    label = "nucleus"

                    if shape.value == "point":
                        draw_centre(img, x1, y1, x2, y2, label, None, cell=False)
                    elif shape.value == "box":
                        draw_box(img, x1, y1, x2, y2, label, None, cell=False)
                    else:
                        raise ValueError(f"No such draw shape {shape.value}")

                print(f"saving to: {save_path}")
            else:
                print(f"no predictions on val_{idx}.png")

            cv2.imwrite(str(save_path), img)


if __name__ == "__main__":
    typer.run(main)
