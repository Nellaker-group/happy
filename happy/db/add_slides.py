import os
from pathlib import Path
from typing import Optional

import typer

from happy.db.slides import Slide, Lab
import happy.db.eval_runs_interface as db


def main(
    slides_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    lab_country: str = typer.Option(...),
    primary_contact: str = typer.Option(...),
    slide_file_format: str = typer.Option(...),
    pixel_size: Optional[float] = None,
    has_notes: bool = False,
    has_clinical_data: bool = False,
):
    """Add a whole lab of slides to the database

    Args:
        slides_dir: absolute path to the dir containing the slides
        lab_country: country where the lab is
        primary_contact: first name of collaborator
        slide_file_format: file format of slides, e.g. '.svs'
        pixel_size: pixel size of all slides. Can be found with QuPath on one slide
        has_notes: if the slides came with associated pathologist's notes
        has_clinical_data: if the slides came with associated clinical data/history
    """
    db.init()

    lab = Lab.create(
        country=lab_country,
        primary_contact=primary_contact,
        slides_dir=slides_dir,
        has_pathologists_notes=has_notes,
        has_clinical_data=has_clinical_data,
    )

    for filename in os.listdir(slides_dir):
        if filename.endswith(slide_file_format):
            Slide.create(slide_name=filename, pixel_size=pixel_size, lab=lab)


if __name__ == "__main__":
    typer.run(main)
