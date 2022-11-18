from pathlib import Path
import pandas as pd
import typer

import happy.db.eval_runs_interface as db
from happy.organs import get_organ


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    slide_name: str = typer.Option(...),
    nuclei_only: bool = False,
):
    """Saves model predictions from database into a tsv that QuPath can read

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project directory
        run_id: id of the run which generated the predictions
        slide_name: shorthand name of the slide for naming the tsv file
        nuclei_only: a flag for when there are no cell predictions in the db
    """
    db.init()

    projects_dir = Path(__file__).parent.parent / "projects"
    save_dir = projects_dir / project_name / "results" / "tsvs"
    organ = get_organ(organ_name)

    save_path = save_dir / f"{slide_name}.tsv"
    coords, preds = _get_db_predictions(run_id)

    coord_to_tsv(coords, preds, save_path, organ, nuclei_only)


def coord_to_tsv(coords, preds, save_path, organ, nuclei_only=False):
    xs = [coord["x"] for coord in coords]
    ys = [coord["y"] for coord in coords]
    if not nuclei_only:
        cell_class = [
            organ.cells[pred["cell_class"]].label if pred is not None else "NA"
            for pred in preds
        ]
    else:
        cell_class = ["Nuclei" for _ in coords]

    print(f"saving {len(preds)} cells to tsv")
    df = pd.DataFrame({"x": xs, "y": ys, "class": cell_class})
    df.to_csv(save_path, sep="\t", index=False)


def _get_db_predictions(run_id):
    coords = db.get_predictions(run_id)
    return coords, coords


if __name__ == "__main__":
    typer.run(main)
