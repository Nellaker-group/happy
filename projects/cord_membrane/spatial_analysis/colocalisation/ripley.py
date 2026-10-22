import numpy as np
from scipy.spatial import cKDTree

from projects.cord_membrane.spatial_analysis.geometry import random_points_in_triangles


def cross_k(reference_points, target_points, area, radii):
    """Ripley's cross-K function: how many reference cells sit around each target cell.

    K(d) = area / (n_target * n_reference) * number of reference-target pairs
    closer than d.

    Args:
        reference_points: (N, 2) coordinates, e.g. leukocytes
        target_points: (M, 2) coordinates of the target cell type
        area: area of the region the cells lie in, in units of the points squared
        radii: distances to evaluate, in units of the points
    Returns:
        (len(radii),) K values in units of the points squared (NaN if either set is empty)
    """
    radii = np.asarray(radii, dtype=np.float64)
    if len(reference_points) == 0 or len(target_points) == 0:
        return np.full(len(radii), np.nan)
    pairs = cKDTree(target_points).count_neighbors(cKDTree(reference_points), radii)
    return area * pairs / (len(target_points) * len(reference_points))


def random_scatter_k(
    target_points, num_reference, triangles, areas, radii, num_simulations, rng
):
    """Cross-K with the reference cells scattered at random over the tissue triangles.

    Returns:
        (num_simulations, len(radii)) K values, one row per simulation
    """
    area = areas.sum()
    return np.array(
        [
            cross_k(
                random_points_in_triangles(triangles, areas, num_reference, rng),
                target_points,
                area,
                radii,
            )
            for _ in range(num_simulations)
        ]
    )
