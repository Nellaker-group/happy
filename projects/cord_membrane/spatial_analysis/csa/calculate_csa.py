import numpy as np
import shapely

from projects.cord_membrane.graph_utils import label_components
from projects.cord_membrane.postprocessing.cord.reclass_tissue import VESSELS
from projects.cord_membrane.spatial_analysis.geometry import (
    alpha_shape,
    diameters,
    fill_holes,
    largest_part,
    within_boundary_ellipse,
)
from projects.cord_membrane.spatial_analysis.csa.calculate_wagenworth import (
    calculate_wagenworth,
    measure_lumen,
)
from projects.cord_membrane.spatial_analysis.csa.calculate_wall_thickness import (
    calculate_wall_thickness,
)


def get_sections(edge_index, num_nodes, min_section_size):
    """Node indices of each cord section, a connected component of the cell graph."""
    labels = label_components(edge_index, num_nodes)
    sizes = np.bincount(labels)
    return [np.flatnonzero(labels == s) for s in np.flatnonzero(sizes >= min_section_size)]


def get_structures(
    edge_index, tissue_predictions, organ, tissue_labels, min_structure_size
):
    """Node indices of each connected structure (e.g. one vessel) for each tissue label."""
    structures = {}
    for label in tissue_labels:
        mask = tissue_predictions == organ.tissue_by_label(label).id
        nodes = np.flatnonzero(mask)
        components = label_components(edge_index, len(tissue_predictions), mask)[nodes]
        ids, sizes = np.unique(components, return_counts=True)
        structures[label] = [
            nodes[components == i] for i in ids[sizes >= min_structure_size]
        ]
    return structures


def measure_areas(
    coords, sections, structures, alpha_radius, section_alpha_radius, pixel_size
):
    """Outline and area of each cord section and of each structure inside it.

    A section is measured on the largest part of its outline. Outlines have their
    holes filled, so a vessel's area includes its lumen. Wharton's jelly is the
    section outline minus all measured structures, so it combines the three
    Wharton's jelly tissues and also contains the amniotic epithelium and any
    structure types that were not measured.

    Args:
        coords: (N, 2) cell coordinates in pixels
        sections: node indices per cord section (see get_sections)
        structures: tissue label -> node indices per structure (see get_structures)
        alpha_radius: alpha shape radius for structures, in pixels
        section_alpha_radius: alpha shape radius for section outlines, in pixels
        pixel_size: microns per pixel
    Returns:
        list of dicts with section, tissue, index, cells, area_um2, outline (holes
        filled), shape (holes kept) and nodes; sections also have perimeter_um,
        major_diameter_um, minor_diameter_um, eccentricity and area_perimeter_ratio
    """
    node_section = np.full(len(coords), -1)
    for i, nodes in enumerate(sections):
        node_section[nodes] = i

    measurements = []
    for i, section_nodes in enumerate(sections):
        section = _measurement(
            i + 1,
            "Section",
            1,
            section_nodes,
            coords,
            section_alpha_radius,
            pixel_size,
            largest_only=True,
        )
        section.update(_shape_measures(section, pixel_size))
        measurements.append(section)

        structure_outlines = []
        structure_cells = 0
        for tissue, components in structures.items():
            in_section = [nodes for nodes in components if node_section[nodes[0]] == i]
            for j, nodes in enumerate(in_section, start=1):
                structure = _measurement(
                    i + 1, tissue, j, nodes, coords, alpha_radius, pixel_size
                )
                structure_outlines.append(structure["outline"])
                structure_cells += len(nodes)
                measurements.append(structure)

        jelly_outline = section["outline"].difference(
            shapely.union_all(structure_outlines)
        )
        measurements.append(
            {
                "section": i + 1,
                "tissue": "WhJelly",
                "index": 1,
                "cells": len(section_nodes) - structure_cells,
                "area_um2": jelly_outline.area * pixel_size**2,
                "outline": jelly_outline,
                "shape": jelly_outline,
                "nodes": None,
            }
        )
    return measurements


def measure_vessels(
    measurements,
    coords,
    cell_predictions,
    organ,
    pixel_size,
    min_endothelial_cells,
    endothelial_ellipse_scale=1.25,
):
    """Lumen, wall area, Wagenworth index and wall thickness of each measured vessel.

    The lumen is the largest hole in the vessel shape, or an ellipse fitted to
    its endothelial cells if the lumen has collapsed (see measure_lumen). The
    wall area S_wa is the vessel area (lumen filled) minus the lumen area S_la.
    Endothelial cells outside their fitted ellipse enlarged by
    endothelial_ellipse_scale are left out as stray calls in the vessel wall.

    Args:
        measurements: output of measure_areas
        coords: (N, 2) cell coordinates in pixels
        cell_predictions: (N,) cell class ids, after reclass_cell if used
        organ: placenta_cord organ
        pixel_size: microns per pixel
        min_endothelial_cells: minimum endothelial cells for the ellipse and wall thickness
    Returns:
        list of dicts, one per vessel, including the lumen outline
    """
    is_endothelial = cell_predictions == organ.cell_by_label("VEN").id
    vessels = []
    for m in measurements:
        if m["tissue"] not in VESSELS:
            continue
        endothelial_points = coords[m["nodes"][is_endothelial[m["nodes"]]]]
        if len(endothelial_points) >= min_endothelial_cells:
            endothelial_points = endothelial_points[
                within_boundary_ellipse(endothelial_points, endothelial_ellipse_scale)
            ]
        lumen, method = measure_lumen(m["shape"], endothelial_points, min_endothelial_cells)
        lumen = lumen.intersection(m["outline"])
        lumen_area = lumen.area * pixel_size**2
        wall_area = m["area_um2"] - lumen_area
        thickness = calculate_wall_thickness(
            m["outline"], endothelial_points, min_endothelial_cells
        )
        vessels.append(
            {
                "section": m["section"],
                "tissue": m["tissue"],
                "index": m["index"],
                "cells": m["cells"],
                "endothelial_cells": len(endothelial_points),
                "vessel_area_um2": m["area_um2"],
                "lumen_area_um2": lumen_area,
                "wall_area_um2": wall_area,
                "lumen_method": method,
                "wagenworth_index": calculate_wagenworth(wall_area, lumen_area),
                "wall_thickness_um": thickness * pixel_size,
                "lumen_outline": lumen,
            }
        )
    return vessels


def _measurement(
    section, tissue, index, nodes, coords, alpha_radius, pixel_size, largest_only=False
):
    shape = alpha_shape(coords[nodes], alpha_radius)
    outline = fill_holes(shape)
    if largest_only:
        outline = largest_part(outline)
    return {
        "section": section,
        "tissue": tissue,
        "index": index,
        "cells": len(nodes),
        "area_um2": outline.area * pixel_size**2,
        "outline": outline,
        "shape": shape,
        "nodes": nodes,
    }


def _shape_measures(measurement, pixel_size):
    perimeter = measurement["outline"].length * pixel_size
    major, minor = np.array(diameters(measurement["outline"])) * pixel_size
    return {
        "perimeter_um": perimeter,
        "major_diameter_um": major,
        "minor_diameter_um": minor,
        "eccentricity": major / minor if minor > 0 else np.nan,
        "area_perimeter_ratio": measurement["area_um2"] / perimeter if perimeter > 0 else np.nan,
    }
