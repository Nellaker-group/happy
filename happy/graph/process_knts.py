import sklearn.neighbors as sk
import numpy as np

from happy.utils.hdf5 import filter_by_cell_type


def process_knt_cells(
    all_predictions,
    all_embeddings,
    all_coords,
    all_confidence,
    organ,
    radius,
    cut_off_count,
    verbose=True,
):
    """Process placenta KNT cells.
    Clusters KNT cells into a single point if they are within a certain radius of
    each other. Relabels isolated KNT cells into SYN cells.
    """

    # Sort by coordinates first to match up with any tissue predictions
    sort_args = np.lexsort((all_coords[:, 1], all_coords[:, 0]))
    all_coords = all_coords[sort_args]
    all_embeddings = all_embeddings[sort_args]
    all_predictions = all_predictions[sort_args]
    all_confidence = all_confidence[sort_args]

    # Filter by KNT cell type
    predictions, embeddings, coords, confidence = filter_by_cell_type(
        all_predictions, all_embeddings, all_coords, all_confidence, "KNT", organ
    )
    if verbose:
        print(f"Data loaded with {len(predictions)} KNT cells")

    # Get indices of KNT cells in radius
    all_knt_inds = np.nonzero(all_predictions == 10)[0]
    unique_indices = _get_indices_in_radius(coords, radius)
    if len(unique_indices) <= 1:
        return all_predictions, all_embeddings, all_coords, all_confidence, np.array([])

    # find KNT cells with no neighbors and turn them into SYN
    all_predictions, predictions = _convert_isolated_knt_into_syn(
        all_predictions,
        predictions,
        cut_off_count,
        unique_indices,
        all_knt_inds,
        verbose=verbose,
    )

    # remove points with more than cut off neighbors and keep just one point
    (
        predictions,
        embeddings,
        coords,
        confidence,
        unique_inds_to_remove,
    ) = _cluster_knts_into_point(
        predictions, embeddings, coords, confidence, cut_off_count, unique_indices
    )

    inds_to_remove_from_total = all_knt_inds[unique_inds_to_remove]
    all_predictions = np.delete(all_predictions, inds_to_remove_from_total, axis=0)
    all_embeddings = np.delete(all_embeddings, inds_to_remove_from_total, axis=0)
    all_coords = np.delete(all_coords, inds_to_remove_from_total, axis=0)
    all_confidence = np.delete(all_confidence, inds_to_remove_from_total, axis=0)
    if verbose:
        print(
            f"Clustered {len(inds_to_remove_from_total)} KNT cells into a single point"
        )

    return (
        all_predictions,
        all_embeddings,
        all_coords,
        all_confidence,
        inds_to_remove_from_total,
    )


def _get_indices_in_radius(coords, radius):
    tree = sk.KDTree(coords, metric="euclidean")
    all_nn_indices = tree.query_radius(coords, r=radius)
    # find indices of duplicate entries and remove duplicates
    unique_indices = np.unique(
        np.array([tuple(row) for row in all_nn_indices], dtype=object)
    )
    return unique_indices


def _convert_isolated_knt_into_syn(
    all_predictions,
    knt_predictions,
    cut_off_count,
    unique_indices,
    all_knt_inds,
    verbose=True,
):
    lone_knt_indices = []
    lone_knt_indices_nested = [
        list(x) for x in unique_indices if len(x) <= cut_off_count
    ]
    for nested in lone_knt_indices_nested:
        lone_knt_indices.extend(nested)
    all_predictions[all_knt_inds[lone_knt_indices]] = 3
    knt_predictions[lone_knt_indices] = 3
    if verbose:
        print(
            f"Converted {len(lone_knt_indices)} KNT cells with fewer than "
            f"{cut_off_count+1} neighbours into SYN"
        )
    return all_predictions, knt_predictions


def _cluster_knts_into_point(
    predictions, embeddings, coords, confidence, cut_off_count, unique_indices
):

    remaining_inds = [x for x in unique_indices if len(x) > cut_off_count]
    # take all elements except the first to be removed from grouped KNT cells
    inds_to_remove = []
    for ind in remaining_inds:
        inds_to_remove.extend(list(ind[1:]))
    # remove duplicates inds
    inds_to_remove = np.array(inds_to_remove)
    unique_inds_to_remove = np.unique(inds_to_remove)

    # remove clustered
    predictions = np.delete(predictions, unique_inds_to_remove, axis=0)
    embeddings = np.delete(embeddings, unique_inds_to_remove, axis=0)
    coords = np.delete(coords, unique_inds_to_remove, axis=0)
    confidence = np.delete(confidence, unique_inds_to_remove, axis=0)
    return predictions, embeddings, coords, confidence, unique_inds_to_remove
