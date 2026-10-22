import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def label_components(edge_index, num_nodes, node_mask=None):
    """Connected component label for each node of the cell graph.

    Components are labelled in order of their lowest node index, so cells
    sorted by x coordinate give components numbered from left to right.

    Args:
        edge_index: (2, E) cell graph edges
        num_nodes: number of nodes in the graph
        node_mask: (N,) bool; if given only edges between masked nodes are used,
            so unmasked nodes each end up in their own component
    Returns:
        (N,) array of component labels
    """
    src, dst = np.asarray(edge_index)
    if node_mask is not None:
        keep = node_mask[src] & node_mask[dst]
        src, dst = src[keep], dst[keep]
    adjacency = coo_matrix(
        (np.ones(len(src), dtype=bool), (src, dst)), shape=(num_nodes, num_nodes)
    )
    return connected_components(adjacency, directed=False)[1]


def remove_long_edges(edge_index, coords, max_length):
    """Edges no longer than max_length, in the units of coords."""
    src, dst = np.asarray(edge_index)
    lengths = np.linalg.norm(
        coords[src].astype(np.float64) - coords[dst].astype(np.float64), axis=1
    )
    keep = lengths <= max_length
    return np.stack([src[keep], dst[keep]])
