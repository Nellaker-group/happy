import time
from typing import Optional

import typer

from happy.organs import get_organ
from happy.utils.utils import get_device, get_project_dir
from happy.cell_infer import nuclei_infer, cell_infer
import happy.db.eval_runs_interface as db


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    nuc_model_id: Optional[int] = None,
    cell_model_id: Optional[int] = None,
    run_id: Optional[int] = None,
    slide_id: Optional[int] = None,
    nuc_num_workers: int = 20,
    cell_num_workers: int = 16,
    score_threshold: float = 0.4,
    max_detections: int = 500,
    nuc_batch_size: int = 16,
    cell_batch_size: int = 800,
    run_nuclei_pipeline: bool = True,
    run_cell_pipeline: bool = True,
):
    """Runs inference over a WSI for nuclei detection, cell classification, or both.

    Cell classification alone requires nuclei detection to have already been
    performed and validated. Will make a new run_id if there are no nuclei detections,
    otherwise it will pick up an existing run to continue nuclei detection, start cell
    classification or continue cell classification.

    Predictions are saved to the database with every batch of images. The cell
    classification step will also stream embeddings to an hdf5 file.

    Args:
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells
        nuc_model_id: id of the nuclei model for inference
        cell_model_id: id of the cell model for inference
        run_id: id of an existing run or of a new run. If none, will auto increment
        slide_id: id of the WSI. Only optional for cell eval.
        nuc_num_workers: number of workers for parallel processing of nuclei inference
        cell_num_workers: number of workers for parallel processing of cell inference
        score_threshold: nuclei network confidence cutoff for saving predictions
        max_detections: max nuclei detections for saving predictions
        nuc_batch_size: batch size for nuclei inference
        cell_batch_size: batch size for cell inference
        run_nuclei_pipeline: True if you want to perform nuclei detection
        run_cell_pipeline: True if you want to perform cell classification
    """
    device = get_device()
    project_dir = get_project_dir(project_name)

    # Create database connection
    db.init()

    if run_nuclei_pipeline:
        # Start timer for nuclei evaluation
        start = time.time()
        # Perform all nuclei evaluation
        run_id = nuclei_eval_pipeline(
            project_dir,
            nuc_model_id,
            slide_id,
            run_id,
            nuc_num_workers,
            nuc_batch_size,
            score_threshold,
            max_detections,
            device,
        )
        end = time.time()
        print(f"Nuclei evaluation time: {(end - start):.3f}")

    if run_cell_pipeline:
        # Start timer for cell evaluation
        start = time.time()
        # Perform all nuclei evaluation
        cell_eval_pipeline(
            project_dir,
            organ_name,
            cell_model_id,
            run_id,
            cell_batch_size,
            cell_num_workers,
            device,
        )
        end = time.time()
        print(f"Cell evaluation time: {(end - start):.3f}")


def nuclei_eval_pipeline(
    project_dir,
    model_id,
    slide_id,
    run_id,
    num_workers,
    batch_size,
    score_threshold,
    max_detections,
    device,
):
    # Load model weights and push to device
    model = nuclei_infer.setup_model(project_dir, model_id, device)
    # Load datasets and dataloader
    dataloader, pred_saver = nuclei_infer.setup_data(
        slide_id, run_id, model_id, batch_size, overlap=200, num_workers=num_workers
    )
    # Predict nuclei
    nuclei_infer.run_nuclei_eval(
        dataloader, model, pred_saver, device, score_threshold, max_detections
    )
    nuclei_infer.clean_up(pred_saver)
    return pred_saver.id


def cell_eval_pipeline(
    project_dir,
    organ_name,
    model_id,
    run_id,
    batch_size,
    num_workers,
    device,
):
    organ = get_organ(organ_name)
    # Load model weights and push to device
    model = cell_infer.setup_model(
        project_dir, model_id, len(organ.cells), device
    )
    # Load datasets and dataloader
    dataloader, pred_saver = cell_infer.setup_data(
        run_id,
        model_id,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Setup or get path to embeddings hdf5 save location
    embeddings_path = cell_infer.setup_embedding_saving(project_dir, pred_saver.id)
    # Predict cell classes
    cell_infer.run_cell_eval(dataloader, model, pred_saver, embeddings_path, device)
    cell_infer.clean_up(pred_saver)


if __name__ == "__main__":
    typer.run(main)
