import numpy as np
import cv2

from PIL import ImageColor


def draw_centre(img, x1, y1, x2, y2, label_name, organ, cell=True, colour=None):
    x = int((x2 + x1) / 2)
    y = int((y2 + y1) / 2)

    if not cell:
        if not colour:
            colour = (255, 182, 109)
        cv2.circle(img, (x, y), 5, colour, 3)
    else:
        hex_colour = organ.cell_by_label(label_name).colour
        label_coords = (int(x) - 15, int(y) - 10)
        colour = _labels_and_colours(img, label_name, label_coords, hex_colour)
        cv2.circle(img, (x, y), 5, colour, 3)


def draw_box(img, x1, y1, x2, y2, label_name, organ, cell=True):
    box = np.array((x1, y1, x2, y2)).astype(int)

    if not cell:
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 182, 109), thickness=2)
    else:
        hex_colour = organ.cell_by_label(label_name).colour
        label_coords = (box[0], box[1] - 10)
        colour = _labels_and_colours(img, label_name, label_coords, hex_colour)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=colour, thickness=2)


def _labels_and_colours(img, label_name, label_coords, hex_colour):
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, label_name, label_coords, font, 1, (0, 0, 0), 2)
    cv2.putText(img, label_name, label_coords, font, 1, (255, 255, 255), 1)

    rgb_colour = ImageColor.getrgb(hex_colour)
    bgr_colour = (rgb_colour[2], rgb_colour[1], rgb_colour[0])
    return bgr_colour


def group_annotations_by_image(df):
    df = df.groupby("image_path", sort=False, as_index=False).agg(
        {
            "x1": lambda x: x.tolist(),
            "y1": lambda x: x.tolist(),
            "x2": lambda x: x.tolist(),
            "y2": lambda x: x.tolist(),
            "class_name": lambda x: x.tolist(),
        }
    )
    return df
