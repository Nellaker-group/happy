from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def visualize_points(
    organ,
    save_path,
    pos,
    width=None,
    height=None,
    labels=None,
    edge_index=None,
    edge_weight=None,
    colours=None,
    point_size=None,
):
    if colours is None:
        colours_dict = {cell.id: cell.colour for cell in organ.cells}
        colours = [colours_dict[label] for label in labels]

    if point_size is None:
        point_size = 1 if len(pos) >= 10000 else 2

    figsize = calc_figsize(pos, width, height)
    fig = plt.figure(figsize=figsize, dpi=300)

    if edge_index is not None:
        line_collection = []
        for i, (src, dst) in enumerate(edge_index.t().tolist()):
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            line_collection.append((src, dst))
        line_colour = (
            [str(weight) for weight in edge_weight.t()[0].tolist()]
            if edge_weight is not None
            else "grey"
        )
        lc = LineCollection(line_collection, linewidths=0.5, colors=line_colour)
        ax = plt.gca()
        ax.add_collection(lc)
        ax.autoscale()
    plt.scatter(
        pos[:, 0],
        pos[:, 1],
        marker=".",
        s=point_size,
        zorder=1000,
        c=colours,
        cmap="Spectral",
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)


def calc_figsize(pos, width, height):
    if width is None and height is None:
        return 8, 8
    if width == -1 and height == -1:
        pos_width = max(pos[:, 0]) - min(pos[:, 0])
        pos_height = max(pos[:, 1]) - min(pos[:, 1])
        ratio = pos_width / pos_height
        length = ratio * 8
        return length, 8
    else:
        ratio = width / height
        length = ratio * 8
        return length, 8
