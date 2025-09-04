from copy import deepcopy

import sklearn.neighbors as sk
import matplotlib.pyplot as plt
import numpy as np

from happy.hdf5 import HDF5Dataset


def process_knts(organ, hdf5_data, tissues=None, verbose=True):
    # Turn isolated knts into syn and group large knts into one point
    hdf5_data, inds_to_remove = process_knt_cells(
        hdf5_data, organ, 50, 3, plot=False, verbose=verbose
    )
    # Remove points from tissue ground truth as well
    if tissues is not None and len(inds_to_remove) > 0:
        tissues = np.delete(tissues, inds_to_remove, axis=0)
    return hdf5_data, tissues


def process_knt_cells(
    hdf5_data: HDF5Dataset, organ, radius, cut_off_count, plot=False, verbose=True
):
    # Filter by KNT cell type
    knt_hdf5_data = deepcopy(hdf5_data)
    knt_hdf5_data, _ = knt_hdf5_data.filter_by_cell_type("KNT", organ)
    if verbose:
        print(f"Data loaded with {len(knt_hdf5_data.cell_predictions)} KNT cells")

    # Get indices of KNT cells in radius
    all_knt_inds = np.nonzero(hdf5_data.cell_predictions == 10)[0]
    unique_indices = _get_indices_in_radius(knt_hdf5_data.coords, radius)
    if len(unique_indices) <= 1:
        return hdf5_data, np.array([])
    if plot:
        _plot_distances_to_nearest_neighbor([len(x) for x in unique_indices])

    # find KNT cells with no neighbors and turn them into SYN
    all_predictions, knt_predictions = _convert_isolated_knt_into_syn(
        hdf5_data.cell_predictions,
        knt_hdf5_data.cell_predictions,
        cut_off_count,
        unique_indices,
        all_knt_inds,
        verbose=verbose,
    )
    hdf5_data.cell_predictions = all_predictions
    knt_hdf5_data.cell_predictions = knt_predictions

    # remove points with more than cut off neighbors and keep just one point
    knt_hdf5_data, unique_inds_to_remove = _cluster_knts_into_point(
        knt_hdf5_data, cut_off_count, unique_indices
    )

    inds_to_remove_from_total = all_knt_inds[unique_inds_to_remove]
    hdf5_data.cell_predictions = np.delete(
        hdf5_data.cell_predictions, inds_to_remove_from_total, axis=0
    )
    hdf5_data.cell_embeddings = np.delete(
        hdf5_data.cell_embeddings, inds_to_remove_from_total, axis=0
    )
    hdf5_data.coords = np.delete(hdf5_data.coords, inds_to_remove_from_total, axis=0)
    hdf5_data.cell_confidence = np.delete(
        hdf5_data.cell_confidence, inds_to_remove_from_total, axis=0
    )
    if verbose:
        print(
            f"Clustered {len(inds_to_remove_from_total)} KNT cells into a single point"
        )

    return hdf5_data, inds_to_remove_from_total


def _plot_distances_to_nearest_neighbor(num_in_radius):
    plt.hist(num_in_radius, bins=100)
    plt.savefig("plots/num_in_radius_histogram.png")
    plt.clf()


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


def _cluster_knts_into_point(knt_hdf5_data, cut_off_count, unique_indices):
    remaining_inds = [x for x in unique_indices if len(x) > cut_off_count]
    # take all elements except the first to be removed from grouped KNT cells
    inds_to_remove = []
    for ind in remaining_inds:
        inds_to_remove.extend(list(ind[1:]))
    # remove duplicates inds
    inds_to_remove = np.array(inds_to_remove)
    unique_inds_to_remove = np.unique(inds_to_remove)

    # remove clustered
    knt_hdf5_data.cell_predictions = np.delete(
        knt_hdf5_data.cell_predictions, unique_inds_to_remove, axis=0
    )
    knt_hdf5_data.cell_embeddings = np.delete(
        knt_hdf5_data.cell_embeddings, unique_inds_to_remove, axis=0
    )
    knt_hdf5_data.coords = np.delete(
        knt_hdf5_data.coords, unique_inds_to_remove, axis=0
    )
    knt_hdf5_data.cell_confidence = np.delete(
        knt_hdf5_data.cell_confidence, unique_inds_to_remove, axis=0
    )
    return knt_hdf5_data, unique_inds_to_remove
