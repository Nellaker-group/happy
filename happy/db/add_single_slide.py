from pathlib import Path
from typing import Optional

import typer

from happy.db.slides import Slide, Lab
import happy.db.eval_runs_interface as db


def main(
    db_name: str = "main.db",
    filename: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    lab_country: str = typer.Option(...),
    primary_contact: str = typer.Option(...),
    pixel_size: Optional[float] = None,
    has_notes: bool = False,
    has_clinical_data: bool = False,
):
    """Add a single slide to the database

    Args:
        filename: absolute path to the slide to add
        lab_country: country where the lab is
        primary_contact: first name of collaborator
        pixel_size: pixel size of all slides. Can be found with QuPath on one slide
        has_notes: if the slides came with associated pathologist's notes
        has_clinical_data: if the slides came with associated clinical data/history
    """
    db.init(db_name)

    slides_dir = filename.parent
    filename = filename.name

    lab = Lab.get_or_create(
        country=lab_country,
        primary_contact=primary_contact,
        slides_dir=slides_dir,
        has_pathologists_notes=has_notes,
        has_clinical_data=has_clinical_data,
    )

    Slide.create(slide_name=filename, pixel_size=pixel_size, lab=lab[0])


if __name__ == "__main__":
    typer.run(main)
