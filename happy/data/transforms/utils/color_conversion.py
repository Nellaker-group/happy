"""
Colour deconvolution reference:
    [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
    staining by color deconvolution.," Analytical and quantitative
    cytology and histology / the International Academy of Cytology [and]
    American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
"""

import numpy as np
from scipy import linalg
from skimage.util.dtype import img_as_float

from happy.data.transforms.utils.rgb_matrices import *


def get_rgb_matrices():
    return [
        np.array(RBG_FROM_HE_1),
        np.array(RBG_FROM_HE_2),
        np.array(RBG_FROM_HE_3),
        np.array(RBG_FROM_HE_QUPATH_DEFAULT),
        np.array(RBG_FROM_HE_LAB5),
        np.array(RBG_FROM_HE_LAB1),
        np.array(RBG_FROM_HE_LAB1_VIVID),
        np.array(RBG_FROM_HE_LAB2),
    ]


def rgb2he(rgb_img, rgb_matrix):
    """Converts RGB image to Haematoxylin-Eosin (HE) color space.
    Args:
        rgb_img: image in RGB format
        rgb_matrix: matrix for conversion
    Return:
        image in HE format
    """
    rgb_matrix[2, :] = np.cross(rgb_matrix[0, :], rgb_matrix[1, :])
    he_matrix = linalg.inv(rgb_matrix)

    rgb_img = img_as_float(rgb_img, force_copy=True)
    rgb_img += .00001  # make sure there are no zeros
    he_image = np.dot(np.reshape(-np.log(rgb_img), (-1, 3)), he_matrix)
    return np.reshape(he_image, rgb_img.shape)


def he2rgb(he_img, rgb_matrix):
    """Converts Haematoxylin-Eosin (HE) to RGB color space.
    Args:
        he_img: image in HE format
        rgb_matrix: matrix for conversion
    Return:
        image in RGB format
    """
    rgb_matrix[2, :] = np.cross(rgb_matrix[0, :], rgb_matrix[1, :])

    stains = img_as_float(he_img)
    logrgb = np.dot(-np.reshape(stains, (-1, 3)), rgb_matrix)
    rgb_img = np.exp(logrgb)
    np.clip(rgb_img, 0.0, 255.0, out=rgb_img)
    return np.reshape(rgb_img, stains.shape)
