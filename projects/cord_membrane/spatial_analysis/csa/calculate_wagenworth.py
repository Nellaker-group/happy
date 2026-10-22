import numpy as np
import shapely

from projects.cord_membrane.spatial_analysis.geometry import (
    boundary_ellipse,
    largest_hole,
)


def measure_lumen(shape, endothelial_points, min_endothelial_cells):
    """Lumen of a vessel, from the empty space inside its wall.

    The lumen is the largest hole in the vessel's alpha shape. Collapsed lumens
    narrower than about twice the alpha radius leave no hole, so these fall back
    to an ellipse fitted to the vessel's endothelial cells (an approximation).

    Args:
        shape: vessel alpha shape with holes (see alpha_shape)
        endothelial_points: (M, 2) coordinates of the vessel's endothelial cells
        min_endothelial_cells: minimum endothelial cells needed to fit an ellipse
    Returns:
        (lumen outline, method) where method is "hole", "ellipse" or "none"
    """
    hole = largest_hole(shape)
    if not hole.is_empty:
        return hole, "hole"
    if len(endothelial_points) >= min_endothelial_cells:
        return boundary_ellipse(endothelial_points), "ellipse"
    return shapely.Polygon(), "none"


def calculate_wagenworth(wall_area, lumen_area):
    """Wagenworth index S_wa / S_la: vessel wall area over lumen area (NaN if no lumen)."""
    return wall_area / lumen_area if lumen_area > 0 else np.nan
