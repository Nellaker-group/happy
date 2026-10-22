"""Download trained HAPPY model weights from the Hugging Face Hub.

The published `main.db` already registers these models by path relative to the
repo root (`projects/<project>/trained_models/<name>.pt`)
The filenames below match those registry entries:

    placenta (projects/placenta/trained_models/):
        nuc-model-id 1    ->  retinanet_placenta_nuclei_detector.pt
        nuc-model-id 2    ->  yolo_multiorgan_nuclei_detector.pt
        cell-model-id 3   ->  resnet50_placenta_cell_classifier.pt
        tissue-model-id 1 ->  clustergcn_placenta_tissue_classifier.pt

    cord and membrane (projects/cord_membrane/trained_models/):
        cell-model-id 4   ->  resnet50_cord_cell_classifier.pt
        cell-model-id 5   ->  resnet50_membrane_cell_classifier.pt
        tissue-model-id 2 ->  clustergcn_cord_tissue_classifier.pt
        tissue-model-id 3 ->  clustergcn_membrane_tissue_classifier.pt
        (cord and membrane use nuc-model-id 2, which stays in projects/placenta/)

! so don't need to add these to the database after downloading as already there

Examples:
    python -m happy.db.download_models --project-name placenta
    python -m happy.db.download_models --project-name cord_membrane

Only register manually (with `add_model`) if you add extra/custom weights beyond these.

nb - if compute doesn't have internet access, download manually
"""
from pathlib import Path
from typing import List
import shutil

import typer

from happy.utils.utils import get_project_dir

# file in the HF repo -> (project whose trained_models/ main.db registers it in, db id)
MODEL_FILES = {
    "nuclei/retinanet_placenta_nuclei_detector.pt": ("placenta", "nuc-model-id 1"),
    "nuclei/yolo_multiorgan_nuclei_detector.pt": ("placenta", "nuc-model-id 2"),
    "cell/resnet50_placenta_cell_classifier.pt": ("placenta", "cell-model-id 3"),
    "tissue/clustergcn_placenta_tissue_classifier.pt": ("placenta", "tissue-model-id 1"),
    "cell/resnet50_cord_cell_classifier.pt": ("cord_membrane", "cell-model-id 4"),
    "cell/resnet50_membrane_cell_classifier.pt": ("cord_membrane", "cell-model-id 5"),
    "tissue/clustergcn_cord_tissue_classifier.pt": ("cord_membrane", "tissue-model-id 2"),
    "tissue/clustergcn_membrane_tissue_classifier.pt": ("cord_membrane", "tissue-model-id 3"),
}
# files needed to run the full pipeline for each project
PROJECT_FILES = {
    "placenta": [f for f, (project, _) in MODEL_FILES.items() if project == "placenta"],
    "cord_membrane": ["nuclei/yolo_multiorgan_nuclei_detector.pt"]
    + [f for f, (project, _) in MODEL_FILES.items() if project == "cord_membrane"],
}


def main(
    repo_id: str = typer.Option("emrwlkr/HAPPY", help="Hugging Face model repo"),
    project_name: str = typer.Option(
        "placenta", help=f"Models to download: placenta or cord_membrane"
    ),
    filename: List[str] = typer.Option(
        [], help="Download only this file, e.g. cell/resnet50_cord_cell_classifier.pt, instead of whole project (repeatable)"
    ),
    revision: str = typer.Option("main", help="Branch, tag, or commit to pin for reproducibility"),
):
    from huggingface_hub import hf_hub_download

    if not filename and project_name not in PROJECT_FILES:
        raise typer.BadParameter(
            f"project_name must be one of {list(PROJECT_FILES)}, or pass --filename"
        )

    for f in filename or PROJECT_FILES[project_name]:
        # registered files go where main.db expects them, anything else into project_name
        project, model_id = MODEL_FILES.get(f, (project_name, "not in main.db, register with add_model"))
        out_dir = get_project_dir(project) / "trained_models"
        out_dir.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=repo_id, filename=f, revision=revision)
        dest = out_dir / Path(f).name
        shutil.copy(cached, dest)
        print(f"Downloaded {repo_id}/{f}  ->  {dest}  ({model_id})")

    print("\nDone. main.db already registers the files above under the ids shown")


if __name__ == "__main__":
    typer.run(main)
