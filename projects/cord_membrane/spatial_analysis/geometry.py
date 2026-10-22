import numpy as np
import shapely
import shapely.affinity
from scipy.spatial import Delaunay, QhullError


def alpha_shape(points, alpha_radius):
    """Concave outline of a set of cell positions.

    Keeps the Delaunay triangles whose circumradius is below alpha_radius and
    merges them. Smaller radii follow concave edges more tightly; empty regions
    wider than about 2 * alpha_radius become holes or split the shape into parts.

    Args:
        points: (N, 2) cell coordinates
        alpha_radius: maximum triangle circumradius, in the units of points
    Returns:
        shapely Polygon or MultiPolygon, holes included (empty if nothing is kept)
    """
    triangles, _ = alpha_triangles(points, alpha_radius)
    if len(triangles) == 0:
        return shapely.Polygon()
    return shapely.union_all(shapely.polygons(triangles))


def alpha_triangles(points, alpha_radius):
    """Triangles making up the alpha shape (see alpha_shape) and their areas."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.empty((0, 3, 2)), np.empty(0)
    try:
        triangles = points[Delaunay(points).simplices]
    except QhullError:  # fewer than 3 non-collinear points
        return np.empty((0, 3, 2)), np.empty(0)

    a = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    b = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        circumradius = a * b * c / (4 * area)

    keep = (area > 0) & (circumradius < alpha_radius)
    return triangles[keep], area[keep]


def random_points_in_triangles(triangles, areas, num_points, rng):
    """Points spread uniformly at random over a set of triangles."""
    chosen = triangles[rng.choice(len(triangles), size=num_points, p=areas / areas.sum())]
    r1, r2 = rng.random((2, num_points, 1))
    flip = r1 + r2 > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    return chosen[:, 0] + r1 * (chosen[:, 1] - chosen[:, 0]) + r2 * (chosen[:, 2] - chosen[:, 0])


def diameters(shape):
    """Longest distance across a shape, and its width perpendicular to that through its midpoint."""
    if shape.is_empty:
        return np.nan, np.nan
    hull = shapely.get_coordinates(shape.convex_hull)
    distances = np.linalg.norm(hull[:, None] - hull[None], axis=-1)
    i, j = np.unravel_index(distances.argmax(), distances.shape)
    major = distances[i, j]
    midpoint = (hull[i] + hull[j]) / 2
    direction = (hull[j] - hull[i]) / major
    perpendicular = np.array([-direction[1], direction[0]]) * major
    line = shapely.LineString([midpoint - perpendicular, midpoint + perpendicular])
    crossings = shapely.get_coordinates(line.intersection(shape.boundary))
    if len(crossings) < 2:
        return major, np.nan
    minor = np.linalg.norm(crossings[:, None] - crossings[None], axis=-1).max()
    return major, minor


def fill_holes(shape):
    """Outline of a shape with its holes filled in (e.g. a vessel including its lumen)."""
    parts = shape.geoms if hasattr(shape, "geoms") else [shape]
    return shapely.union_all(
        [shapely.Polygon(part.exterior) for part in parts if not part.is_empty]
    )


def largest_part(shape):
    """Largest polygon of a shape, or an empty polygon."""
    parts = shape.geoms if hasattr(shape, "geoms") else [shape]
    return max(parts, key=lambda part: part.area, default=shapely.Polygon())


def largest_hole(shape):
    """Largest hole inside a shape (e.g. a vessel lumen), or an empty polygon."""
    parts = shape.geoms if hasattr(shape, "geoms") else [shape]
    holes = [shapely.Polygon(ring) for part in parts for ring in part.interiors]
    return max(holes, key=lambda hole: hole.area, default=shapely.Polygon())


def boundary_ellipse(points):
    """Ellipse with the same spread as points lying around its boundary.

    For points spread evenly around an ellipse, the variance along each axis is
    half the squared semi-axis, so the semi-axes are sqrt(2 * eigenvalues) of
    the points' covariance.
    """
    points = np.asarray(points, dtype=np.float64)
    centre = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
    minor, major = np.sqrt(2 * np.clip(eigenvalues, 0, None))
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    circle = shapely.Point(centre).buffer(1, quad_segs=32)
    ellipse = shapely.affinity.scale(circle, major, minor)
    return shapely.affinity.rotate(ellipse, angle, origin=tuple(centre))


def within_boundary_ellipse(points, scale=1.0):
    """Mask of points inside their boundary ellipse (see boundary_ellipse) enlarged by scale."""
    points = np.asarray(points, dtype=np.float64)
    centred = points - points.mean(axis=0)
    inverse_covariance = np.linalg.pinv(np.cov(points, rowvar=False))
    distance = np.einsum("ij,jk,ik->i", centred, inverse_covariance, centred)
    return distance <= 2 * scale**2
