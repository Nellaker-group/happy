"""Download trained HAPPY model weights from the Hugging Face Hub.

The published `main.db` already registers these models by relative path
(`trained_models/<name>.pt`)
The default filenames below match those registry entries:

    nuc-model-id 1  ->  retinanet_placenta_nuclei_detector.pt
    nuc-model-id 2  ->  yolo_multiorgan_nuclei_detector.pt
    cell-model-id 3 ->  resnet50_placenta_cell_classifier.pt
    tissue-model-id 1 -> clustergcn_placenta_tissue_classifier.pt

! so don't need to add these to the database after downloading as already there

Example (downloads all four into projects/placenta/trained_models/):
    python -m happy.db.download_models --project-name placenta

Only register manually (with `add_model`) if you add extra/custom weights beyond these.

nb - if compute doesn't have internet access, download manually
"""
from pathlib import Path
from typing import List
import shutil

import typer

from happy.utils.utils import get_project_dir

DEFAULT_FILES = [
    "nuclei/retinanet_placenta_nuclei_detector.pt",
    "nuclei/yolo_multiorgan_nuclei_detector.pt",
    "cell/resnet50_placenta_cell_classifier.pt",
    "tissue/clustergcn_placenta_tissue_classifier.pt",
]


def main(
    repo_id: str = typer.Option("emrwlkr/HAPPY", help="Hugging Face model repo"),
    filename: List[str] = typer.Option(
        DEFAULT_FILES, help="File(s) within the repo to download (repeatable)"
    ),
    project_name: str = typer.Option("placenta", help="Download into projects/<project>/trained_models/"),
    revision: str = typer.Option("main", help="Branch, tag, or commit to pin for reproducibility"),
):
    from huggingface_hub import hf_hub_download

    out_dir = get_project_dir(project_name) / "trained_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in filename:
        cached = hf_hub_download(repo_id=repo_id, filename=f, revision=revision)
        dest = out_dir / Path(f).name
        shutil.copy(cached, dest)
        print(f"Downloaded {repo_id}/{f}  ->  {dest}")

    print(
        "\nDone. main.db references these files by relative path \n"
        "nuclei/retinanet_placenta_nuclei_detector.pt > nuc model id 1 \n"
        "nuclei/yolo_multiorgan_nuclei_detector.pt > nuc model id 2 \n"
        "cell/resnet50_placenta_cell_classifier.pt > cell model id 3 \n"
        "tissue/clustergcn_placenta_tissue_classifier.pt > tissue model id 1 \n"
    )


if __name__ == "__main__":
    typer.run(main)
