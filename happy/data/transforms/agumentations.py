import random

import albumentations as al
import numpy as np

from happy.data.transforms.utils.color_conversion import he2rgb, rgb2he


class AlbAugmenter(object):
    def __init__(
        self,
        list_of_albumentations,
        prgn=42,
        min_area=0.0,
        min_visibility=0.0,
        bboxes=True,
    ):
        self.list_of_albumentations = list_of_albumentations
        self.prgn = prgn
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.bboxes = bboxes

    def __call__(self, sample):
        if self.bboxes:
            sample = self.bound_bbox_within_image(sample)
            alb_format = {
                "image": sample["img"],
                "bboxes": [x[:-1] for x in sample["annot"]],
                "category_id": [x[-1] for x in sample["annot"]],
            }
        else:
            alb_format = {"image": sample["img"], "category_id": sample["annot"]}

        if self.bboxes:
            alb_aug = al.Compose(
                self.list_of_albumentations,
                bbox_params={
                    "format": "pascal_voc",
                    "min_area": self.min_area,
                    "min_visibility": self.min_visibility,
                    "label_fields": ["category_id"],
                },
            )
        else:
            alb_aug = al.Compose(self.list_of_albumentations)
        alb_format = alb_aug(**alb_format)
        if self.bboxes:
            sample = {
                "img": alb_format["image"],
                "annot": np.array(
                    [
                        np.append(y, x)
                        for y, x in zip(
                            alb_format["bboxes"],
                            [[z] for z in alb_format["category_id"]],
                        )
                    ]
                ),
            }
        else:
            sample = {
                "img": alb_format["image"],
                "annot": sample["annot"],
            }
        return sample

    def bound_bbox_within_image(self, sample):
        im_y, im_x, im_channels = sample["img"].shape
        for xindex, x in enumerate(sample["annot"]):
            bbx_x1, bbx_y1, bbx_x2, bbx_y2, bbx_cat = x
            assert bbx_x1 < bbx_x2
            assert bbx_y1 < bbx_y2
            if not bbx_x1 >= 0:
                bbx_x1 = 0
            if not bbx_y1 >= 0:
                bbx_y1 = 0
            if not bbx_x2 <= im_x:
                bbx_x2 = im_x
            if not bbx_y2 <= im_y:
                bbx_y2 = im_y
            sample["annot"][xindex] = [bbx_x1, bbx_y1, bbx_x2, bbx_y2, bbx_cat]
        return sample


class StainAugment(al.ImageOnlyTransform):
    """Convert the input RGB image to HE (Heamatoxylin, Eosin) vary the staining
    in H and E, return to RGB.
    Args:
        rgb_matrices (list): list of array rgb conversion matrices.
        p (float): probability of applying the transform. Default: 0.5.
        variance (float): factor by which HE staining is randomly varied +-
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, rgb_matrices, variance=0.1, always_apply=False, p=0.5):
        super(StainAugment, self).__init__(always_apply, p)
        self.rgb_matrices = rgb_matrices
        self.variance = variance

    def apply(self, img, variance=0.1, **params):
        # Randomly select rbg matrix to use
        rgb_matrix = random.choice(self.rgb_matrices)
        # Convert to he colour space
        img = rgb2he(img * 1.0, rgb_matrix)
        # Randomly vary Heamatoxylin by -+ variance value
        img[:, :, [0]] = (
            np.random.uniform(low=-variance, high=variance) + img[:, :, [0]]
        )
        # Randomly vary Eosin by -+ variance value
        img[:, :, [1]] = (
            np.random.uniform(low=-variance, high=variance) + img[:, :, [1]]
        )
        # Convert back to rgb colour space
        img = he2rgb(img, rgb_matrix)
        return img.astype(np.uint8)

    def get_params(self):
        return {"variance": self.variance}
