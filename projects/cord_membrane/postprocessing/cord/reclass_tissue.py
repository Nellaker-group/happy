from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from happy.organs import get_organ
from projects.cord_membrane.graph_utils import label_components

VESSELS = ["UArtery", "UVein"]
REMNANTS = ["Allantois", "VitDuct", "VitVes"]


def reclass_tissue(
    tissue_predictions, coords, section_edges, vessel_edges, min_vessel_size=500
):
    """Relabel umbilical vessels so vessels are not mixed predictions.

    Vessels are connected groups (over vessel_edges) of cells predicted as UArtery
    or UVein with at least min_vessel_size cells. Within each cord section (a
    connected component over section_edges):
        - 6, 9 or 12 vessels: grouped into cords of three by their centres, then
          relabelled per cord as below
        - 2+ vessels: the largest third by area (at least one) become UVein and
          the rest become UArtery
        - 1 vessel: takes its majority label
    Smaller vessels take the most common non-vessel tissue around them.

    Args:
        tissue_predictions: (N,) tissue class ids for the cord organ
        coords: (N, 2) cell coordinates
        section_edges: (2, E) cell graph edges used to find cord sections
        vessel_edges: (2, E) cell graph edges used to find vessels
        min_vessel_size: minimum number of cells for a vessel to be relabelled
    Returns:
        (N,) array of relabelled tissue class ids
    """
    organ = get_organ("cord")
    artery = organ.tissue_by_label("UArtery").id
    vein = organ.tissue_by_label("UVein").id

    tissue_predictions = np.asarray(tissue_predictions)
    num_nodes = len(tissue_predictions)

    sections = label_components(section_edges, num_nodes)
    is_vessel = np.isin(tissue_predictions, [artery, vein])
    vessels = label_components(vessel_edges, num_nodes, is_vessel)

    # per vessel: size, artery fraction, centre, area and the cord section it belongs to
    vessel_nodes = np.flatnonzero(is_vessel)
    _, node_to_vessel, sizes = np.unique(
        vessels[vessel_nodes], return_inverse=True, return_counts=True
    )
    is_artery = tissue_predictions[vessel_nodes] == artery
    artery_fraction = np.bincount(node_to_vessel, weights=is_artery) / sizes
    centres = np.stack(
        [np.bincount(node_to_vessel, weights=coords[vessel_nodes, i]) / sizes for i in (0, 1)],
        axis=1,
    )
    vessel_section = np.zeros(len(sizes), dtype=np.int64)
    vessel_section[node_to_vessel] = sections[vessel_nodes]
    large = sizes >= min_vessel_size
    areas = np.zeros(len(sizes))
    for v in np.flatnonzero(large):
        areas[v] = _hull_area(coords[vessel_nodes[node_to_vessel == v]])

    section_sizes = np.bincount(sections)
    new_labels = np.full(len(sizes), -1)
    for i, section in enumerate(np.unique(vessel_section[large]), start=1):
        section_vessels = np.flatnonzero(large & (vessel_section == section))
        if len(section_vessels) in (6, 9, 12):
            rule = "largest is vein, per cord"
            for cord in _group_into_cords(centres[section_vessels]):
                _largest_are_veins(section_vessels[cord], areas, new_labels, artery, vein)
        elif len(section_vessels) >= 2:
            rule = "largest is vein"
            _largest_are_veins(section_vessels, areas, new_labels, artery, vein)
        else:
            rule = "majority vote"
            new_labels[section_vessels] = np.where(
                artery_fraction[section_vessels] >= 0.5, artery, vein
            )
        print(
            f"Cord section {i}: {len(section_vessels)} vessels, "
            f"{section_sizes[section]:,} cells ({rule})"
        )

    updated_predictions = tissue_predictions.copy()
    node_labels = new_labels[node_to_vessel]
    relabel = node_labels >= 0
    updated_predictions[vessel_nodes[relabel]] = node_labels[relabel]

    in_small_vessel = np.zeros(num_nodes, dtype=bool)
    in_small_vessel[vessel_nodes] = ~large[node_to_vessel]
    excluded = [organ.tissue_by_label(label).id for label in VESSELS + REMNANTS]
    updated_predictions = relabel_from_neighbours(
        updated_predictions, vessel_edges, in_small_vessel, excluded
    )
    print(
        f"Relabelled {large.sum()} vessels across {len(np.unique(vessel_section[large]))} "
        f"cord section(s) and {in_small_vessel.sum()} cells in smaller vessels, changing "
        f"{np.sum(updated_predictions != tissue_predictions)} cells"
    )
    return updated_predictions


def reclass_remnants(tissue_predictions, edges):
    """Relabel allantoic and vitelline remnants with the most common other tissue around them."""
    organ = get_organ("cord")
    remnants = [organ.tissue_by_label(label).id for label in REMNANTS]
    is_remnant = np.isin(tissue_predictions, remnants)
    updated_predictions = relabel_from_neighbours(
        tissue_predictions, edges, is_remnant, remnants
    )
    print(f"Relabelled {np.sum(updated_predictions != tissue_predictions)} remnant cells")
    return updated_predictions


def relabel_from_neighbours(tissue_predictions, edges, mask, excluded):
    """Give each connected group of masked cells the most common tissue bordering it.

    Bordering cells with an excluded tissue are ignored, and groups with no other
    bordering cells are left unchanged.
    """
    tissue_predictions = np.asarray(tissue_predictions)
    updated_predictions = tissue_predictions.copy()
    groups = label_components(edges, len(tissue_predictions), mask)
    src, dst = np.asarray(edges)
    border = mask[src] & ~mask[dst] & ~np.isin(tissue_predictions[dst], excluded)
    if not border.any():
        return updated_predictions

    votes = pd.DataFrame(
        {"group": groups[src[border]], "tissue": tissue_predictions[dst[border]]}
    )
    best = (
        votes.value_counts()
        .reset_index()
        .drop_duplicates("group")
        .set_index("group")["tissue"]
    )
    nodes = np.flatnonzero(mask)
    new_tissue = best.reindex(groups[nodes]).to_numpy()
    found = ~pd.isna(new_tissue)
    updated_predictions[nodes[found]] = new_tissue[found].astype(updated_predictions.dtype)
    return updated_predictions


def _largest_are_veins(vessels, areas, labels, artery, vein):
    by_area = vessels[np.argsort(-areas[vessels], kind="stable")]
    num_veins = max(1, len(vessels) // 3)
    labels[by_area[:num_veins]] = vein
    labels[by_area[num_veins:]] = artery


def _hull_area(points):
    try:
        return ConvexHull(np.asarray(points, dtype=np.float64)).volume
    except QhullError:
        return 0.0


def _group_into_cords(centres):
    # groups of three vessels with the smallest total distance to their group centre
    best_cost, best_groups = np.inf, None

    def search(remaining, groups, cost):
        nonlocal best_cost, best_groups
        if cost >= best_cost:
            return
        if not remaining:
            best_cost, best_groups = cost, groups
            return
        first, rest = remaining[0], remaining[1:]
        for pair in combinations(rest, 2):
            group = [first, *pair]
            points = centres[group]
            group_cost = np.linalg.norm(points - points.mean(axis=0), axis=1).sum()
            search([v for v in rest if v not in pair], groups + [group], cost + group_cost)

    search(list(range(len(centres))), [], 0.0)
    return best_groups
