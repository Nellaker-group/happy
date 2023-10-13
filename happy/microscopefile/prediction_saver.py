import numpy as np
import sklearn.neighbors as sk

import happy.db.eval_runs_interface as db


class PredictionSaver:
    """Class for saving model predictions on a whole slide image (WSI).

    Data is saved and read from a DB by the public interface. Saved coordinates match
    the original WSI. This means predictions are scaled to the pixel size of the
    original WSI. Before getting predictions, coords are scaled up to model pixel sizes.

    Args:
        file: MicroscopeFile object
    """

    def __init__(self, microscopefile):
        self.file = microscopefile
        self.id = self.file.id
        self.rescale_ratio = self.file.rescale_ratio

    # Saving tiles which do not have predictions as caught by pixel colours
    def save_empty(self, tile_indexes):
        db.mark_finished_tiles(self.id, tile_indexes)

    # Saves center location of box predictions in terms of whole slide coordinates
    def save_nuclei(self, tile_index, boxes):
        tile_x = str(self.file.tile_xy_list[tile_index][0])
        tile_y = str(self.file.tile_xy_list[tile_index][1])

        cell_tile_size = 200 * self.file.rescale_ratio

        coords = []
        if boxes.size == 0:
            db.mark_finished_tiles(self.id, [tile_index])
        else:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                centroid_x = int(x1 + x2) / 2
                centroid_y = int(y1 + y2) / 2
                centroid_x *= self.rescale_ratio
                centroid_y *= self.rescale_ratio

                centroid_x += int(tile_x)
                centroid_y += int(tile_y)

                # Only saves predictions if they are inside WSI dimensions
                if (
                    centroid_x > 0
                    and centroid_y > 0
                    and (centroid_x + cell_tile_size) < self.file.max_slide_width
                    and (centroid_y + cell_tile_size) < self.file.max_slide_height
                ):
                    coords.append((centroid_x, centroid_y))

            db.save_pred_workings(self.id, coords)
            db.mark_finished_tiles(self.id, [tile_index])

    # Cluster overlapped tiles. Overlap value results in multiple predictions for
    # same nuc, this removes them.
    def cluster_multi_detections(self, nuclei_preds, dist_threshold=4):
        print("finding duplicate nuclei clusters to cluster into one")
        tree = sk.KDTree(nuclei_preds, metric="euclidean")

        # each element contains index of point and index of neighbours within radius
        all_nn_indices = tree.query_radius(nuclei_preds, r=dist_threshold)

        # find all inds with at least one neighbour within radius and fewer than 5
        dup_det_inds = [x for x in all_nn_indices if 1 < len(x) < 5]

        # remove some identical entries
        # (i.e. if there are two neighbours there will be 2 entries)
        unique_dup_det_inds = np.unique(
            np.array([tuple(row) for row in dup_det_inds], dtype=object)
        )

        # if all elements of dup_det_inds were size 2, then you need a different unique
        if not isinstance(unique_dup_det_inds[0], tuple):
            unique_dup_det_inds = np.unique(dup_det_inds, axis=0)

        # take all elements except the first to be removed from original nuc_loc
        inds_to_remove = []
        for ind in unique_dup_det_inds:
            inds_to_remove.extend(list(ind[1:]))

        # remove duplicates inds
        inds_to_remove = np.array(inds_to_remove)
        unique_inds_to_remove = np.unique(inds_to_remove)

        # remove clustered
        coords = np.delete(nuclei_preds, unique_inds_to_remove, axis=0)

        print(
            f"nuclei clustering: {len(unique_inds_to_remove)} "
            f"duplicate nuclei predictions found and marked as invalid"
        )
        return coords

    def remove_edge_nuclei(self, nuclei_predictions, length):
        print("finding edge nuclei to remove")
        l = length * self.rescale_ratio
        max_w = self.file.max_slide_width
        max_h = self.file.max_slide_height
        nuc_loc = nuclei_predictions

        mask = np.logical_and(
            (np.logical_and((nuc_loc[:, 0] - l) > 0, ((nuc_loc[:, 1] - l) > 0))),
            (
                np.logical_and(
                    (nuc_loc[:, 0] + l) < max_w, ((nuc_loc[:, 1] + l) < max_h)
                )
            ),
        )

        valid_nuclei = nuc_loc[mask]

        print(
            f"edge nuclei: {len(nuc_loc) - len(valid_nuclei)} "
            f"edge nuclei found and marked as invalid"
        )
        return valid_nuclei

    def apply_nuclei_post_processing(self, cluster=True, remove_edges=True):
        nuclei_preds = db.get_all_unvalidated_nuclei_preds(self.id)
        nuclei_preds = np.array(list(zip(nuclei_preds["x"], nuclei_preds["y"])))
        if cluster:
            nuclei_preds = self.cluster_multi_detections(nuclei_preds)
        if remove_edges:
            nuclei_preds = self.remove_edge_nuclei(nuclei_preds, 200)
        # save to db and mark nuclei as valid
        nuclei_preds = nuclei_preds.tolist()
        db.validate_pred_workings(self.id, nuclei_preds)
        self.file.mark_finished_nuclei()

    # Inserts valid/non duplicate predictions into Predictions table
    def commit_valid_nuclei_predictions(self):
        db.commit_pred_workings(self.id)

    # Saves cell class predictions at given nuclei coords
    def save_cells(self, coords, predictions):
        db.save_cells(self.id, coords, predictions)

    def finished_cells(self):
        self.file.mark_finished_cells()

    @staticmethod
    def filter_by_score(max_detections, threshold, scores, boxes):
        indices = np.where(scores > threshold)[0]
        if indices.shape[0] > 0:
            # select those scores
            scores = scores[indices]
            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]
            # select detections
            return boxes[indices[scores_sort], :]
        else:
            return np.array([])
