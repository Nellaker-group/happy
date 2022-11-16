import cv2
import numpy as np
import skimage.transform
from PIL import Image, ImageChops


class Resizer(object):
    """Scales to size model was trained on.
    If model wasn't trained with a resizer then eval won't need one either"""

    def __init__(
        self, min_side=608, max_side=1024, padding=True, scale_annotations=True
    ):
        self.min_side = min_side
        self.max_side = max_side
        self.padding = padding
        self.scale_annotations = scale_annotations

    def __call__(self, sample):
        image, annots = sample["img"], sample['annot']
        if not isinstance(annots, int):
            annots = annots.astype(np.float64)

        rows, cols, _ = image.shape

        min_side_scale, max_side_scale = _compute_scale(
            rows, cols, self.min_side, self.max_side
        )

        # This is only for nuc_detection. The scales should be equal here anyway
        if self.scale_annotations and not len(annots) == 0:
            assert min_side_scale == max_side_scale
            annots[:, :4] *= max_side_scale

        sample.update(
            {
                "img": self._resize_image(image, min_side_scale, max_side_scale),
                "annot": annots,
                "scale": max_side_scale,
            }
        )
        return sample

    def _resize_image(self, image, min_side_scale, max_side_scale):
        rows, cols, _ = image.shape

        if rows > cols:
            scaled_row_size = int(round(rows * max_side_scale))
            scaled_col_size = int(round(cols * min_side_scale))
        else:
            scaled_row_size = int(round(rows * min_side_scale))
            scaled_col_size = int(round(cols * max_side_scale))

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image, (scaled_row_size, scaled_col_size), mode="constant"
        )  # added mode to supress warning
        rows, cols, cns = image.shape

        if self.padding:
            pad_w = 32 - rows % 32
            pad_h = 32 - cols % 32
        else:
            pad_w = 0
            pad_h = 0

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return new_image


def _compute_scale(rows, cols, min_side, max_side):
    # rescale image both ways to make a square
    if max_side == min_side:
        min_side_scale = min_side / min(rows, cols)
        max_side_scale = max_side / max(rows, cols)
        return min_side_scale, max_side_scale
    else:
        # rescale the image so the smallest side is min_side
        min_side_scale = min_side / min(rows, cols)
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * min_side_scale > max_side:
            max_side_scale = max_side / largest_side
        else:
            max_side_scale = min_side_scale
        return min_side_scale, max_side_scale


class Normalizer(object):
    def __init__(
        self,
        mean=np.array([[[0.485, 0.456, 0.406]]]),
        std=np.array([[[0.229, 0.224, 0.225]]]),
    ):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["img"]
        # Normalising requires image to be float32 and returns the image as float32
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        sample.update({"img": _normalise_image(image, self.mean, self.std)})
        return sample


def _normalise_image(image, mean, std):
    return (image.astype(np.float32) - mean) / std


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if not mean:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if not std:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def unnormalise_image(image):
    unnorm = UnNormalizer()
    img = unnorm(image).permute(1, 2, 0).numpy()
    # Converts the image back to uin8 to work with image saving/visualisation code
    img = img * 255
    return Image.fromarray(img.astype("uint8"))


def untransform_image(image):
    # unnormalise
    img = unnormalise_image(image)

    # remove padding
    bg = Image.new(img.mode, img.size, img.getpixel((img.size[0] - 1, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    img = img.crop(diff.getbbox())

    # resize
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, dsize=(1600, 1200))
    return img
