"""
Collators deal with transformations to whole batches
"""

import numpy as np
import torch


# Converts to tensors
def cell_collater(batch):
    # take all possible keys
    sample_dict = {}
    for key in batch[0].keys():
        if key != "img" or key != "annot":
            sample_dict[key] = [data[key] for data in batch]

    imgs = np.array([data["img"] for data in batch])
    annots = np.array([data["annot"] for data in batch])

    # (batch, channel, x, y) for resnet that's (100, 224, 244, 3)
    transposed_imgs = np.transpose(imgs, axes=[0, 3, 2, 1])

    sample_dict["img"] = torch.FloatTensor(transposed_imgs)
    sample_dict["annot"] = torch.LongTensor(annots)

    return sample_dict


# adds padding to images and annotations. This is only needed if model was trained with
# this collater when used for eval on a microscopefiles, this checks for empty tiles
# and does not pass them to the model. When used for training, this checks for training
# images without annotations
def collater(batch):
    # take all possible keys
    sample_dict = {}
    for key in batch[0].keys():
        sample_dict[key] = [data[key] for data in batch]

    imgs = [
        torch.from_numpy(data["img"]) if data["img"] is not None else None
        for data in batch
    ]
    annots = [
        torch.from_numpy(data["annot"]) if data["annot"] is not None else None
        for data in batch
    ]
    batch_size = len(imgs)

    # If all images in the batch are empty
    if all(img is None for img in imgs):
        padded_imgs = [None for _ in range(batch_size)]
        annot_padded = [None for _ in range(batch_size)]
        sample_dict.update({"img": padded_imgs, "annot": annot_padded})
    else:
        # There is at least one non-empty tile in the batch
        # find max dimensions of images is it isn't empty
        max_width = np.max([int(img.shape[0]) for img in imgs if img is not None])
        max_height = np.max([int(img.shape[1]) for img in imgs if img is not None])

        # pad images with 0 so all dimensions match and some extra padding of 3.
        padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

        for i in range(batch_size):
            # replace the padding locations with image values if the image isn't empty
            if imgs[i] is not None:
                padded_imgs[
                    i, : int(imgs[i].shape[0]), : int(imgs[i].shape[1]), :
                ] = imgs[i]

        padded_imgs = padded_imgs.permute(0, 3, 1, 2)  # channel, x, y

        # find max number of annotations in batch
        max_num_annots = np.max([annot.shape[0] for annot in annots])

        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

            for i, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[i, : annot.shape[0], :] = annot
        else:
            # this is changes the empty annotations to [-1,-1,-1,-1,-1]
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        sample_dict.update({"img": padded_imgs, "annot": annot_padded})

    return sample_dict
