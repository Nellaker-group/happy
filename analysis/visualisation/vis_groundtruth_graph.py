import typer
import numpy as np

from happy.graph.create_graph import get_groundtruth_patch
from happy.utils.utils import get_project_dir
from happy.organs import get_organ
from happy.graph.visualise import visualize_points


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    annot_tsv: str = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    remove_unlabelled: bool = False,
    lab_id: int = typer.Option(...),
    slide_name: str = typer.Option(...),
):
    """Visualises the ground truth points within a region for the graph

    Args:
        project_name: name of the project
        organ_name: name of the organ to get the cell colours
        annot_tsv: name of the annotation tsv file with ground truth points
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        remove_unlabelled: whether to exclude unlabelled points
        lab_id: id of the lab
        slide_name: name of the slide
    """

    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    xs, ys, tissue_class = get_groundtruth_patch(
        organ, project_dir, x_min, y_min, width, height, annot_tsv
    )

    if remove_unlabelled:
        labelled_inds = tissue_class.nonzero()
        tissue_class = tissue_class[labelled_inds]
        xs = xs[labelled_inds]
        ys = ys[labelled_inds]

    unique, counts = np.unique(tissue_class, return_counts=True)
    print(dict(zip(unique, counts)))

    save_dir = (
        project_dir
        / "visualisations"
        / "graphs"
        / f"lab_{lab_id}"
        / slide_name
        / "groundtruth"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}"
    save_path = save_dir / plot_name

    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in tissue_class]
    visualize_points(
        organ,
        save_path,
        np.stack((xs, ys), axis=1),
        colours=colours,
        width=width,
        height=height,
    )


if __name__ == "__main__":
    typer.run(main)
