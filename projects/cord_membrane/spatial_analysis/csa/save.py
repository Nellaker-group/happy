import json

import pandas as pd
import shapely.geometry

AREA_COLUMNS = ["section", "tissue", "index", "cells", "area_um2"]
SECTION_COLUMNS = [
    "section",
    "cells",
    "area_um2",
    "perimeter_um",
    "major_diameter_um",
    "minor_diameter_um",
    "eccentricity",
    "area_perimeter_ratio",
]
VESSEL_COLUMNS = [
    "section",
    "tissue",
    "index",
    "cells",
    "endothelial_cells",
    "vessel_area_um2",
    "lumen_area_um2",
    "wall_area_um2",
    "lumen_method",
    "wagenworth_index",
    "wall_thickness_um",
]


def save_areas(save_path, measurements):
    """Save one row per cord section, structure and Wharton's jelly region."""
    _save_csv(save_path, measurements, AREA_COLUMNS)
    print(f"Saved areas to {save_path}")


def save_sections(save_path, measurements):
    """Save one row per cord section with its area, perimeter, diameters and eccentricity."""
    sections = [m for m in measurements if m["tissue"] == "Section"]
    _save_csv(save_path, sections, SECTION_COLUMNS)
    print(f"Saved section measurements to {save_path}")


def save_vessels(save_path, vessels):
    """Save one row per vessel with its lumen, wall, Wagenworth index and wall thickness."""
    _save_csv(save_path, vessels, VESSEL_COLUMNS)
    print(f"Saved vessel measurements to {save_path}")


def save_qupath_outlines(save_path, measurements, vessels, organ):
    """Save outlines and vessel lumens as GeoJSON annotations, which can be dragged into QuPath."""
    colours = {tissue.label: tissue.colour for tissue in organ.tissues}
    features = [
        _feature(
            m["outline"],
            f"Section {m['section']} {m['tissue']} {m['index']}",
            m["tissue"],
            colours.get(m["tissue"]),
        )
        for m in measurements
    ]
    features += [
        _feature(
            v["lumen_outline"],
            f"Section {v['section']} {v['tissue']} {v['index']} lumen",
            "Lumen",
            None,
        )
        for v in vessels
    ]
    features = [feature for feature in features if feature is not None]
    with open(save_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"Saved QuPath outlines to {save_path}")


def _save_csv(save_path, rows, columns):
    rows = [{column: row[column] for column in columns} for row in rows]
    pd.DataFrame(rows, columns=columns).to_csv(save_path, index=False)


def _feature(outline, name, classification_name, colour):
    if outline.is_empty:
        return None
    classification = {"name": classification_name}
    if colour:
        classification["color"] = _hex_to_rgb(colour)
    return {
        "type": "Feature",
        "geometry": shapely.geometry.mapping(outline),
        "properties": {
            "objectType": "annotation",
            "name": name,
            "classification": classification,
        },
    }


def _hex_to_rgb(colour):
    colour = colour.lstrip("#")
    return [int(colour[i : i + 2], 16) for i in (0, 2, 4)]
