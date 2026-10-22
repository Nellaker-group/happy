import numpy as np
import shapely


def calculate_wall_thickness(outline, endothelial_points, min_endothelial_cells):
    """Mean distance from a vessel's endothelial cells to its outer edge.

    Args:
        outline: vessel outline with the lumen filled (see fill_holes)
        endothelial_points: (M, 2) coordinates of the vessel's endothelial cells
        min_endothelial_cells: minimum endothelial cells needed for a measurement
    Returns:
        mean wall thickness in the units of the points (NaN if too few cells)
    """
    if len(endothelial_points) < min_endothelial_cells:
        return np.nan
    distances = shapely.distance(shapely.points(endothelial_points), outline.boundary)
    return float(np.mean(distances))
