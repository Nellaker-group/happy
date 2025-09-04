import time
import logging
from typing import Optional

import typer

from happy.organs import get_organ
from happy.utils.utils import get_device
from happy.cell_infer import nuclei_infer, cell_infer
import happy.db.eval_runs_interface as db


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    db_name: str = 'main.db',
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
    tile_size: int = 15000,
    run_nuclei_pipeline: bool = True,
    run_cell_pipeline: bool = True,
    re_run_all_cell:bool = False,
    get_cuda_device_num: bool = False,
    verbose: bool = True,
):
    """Runs inference over a WSI for nuclei detection, cell classification, or both.

    Cell classification alone requires nuclei detection to have already been
    performed and validated. Will make a new run_id if there are no nuclei detections,
    otherwise it will pick up an existing run to continue nuclei detection, start cell
    classification or continue cell classification.

    Predictions are saved to the database with every batch.

    Args:
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells
        db_name: name of the database to connect to (default: main.db)
        nuc_model_id: id of the nuclei model for inference
        cell_model_id: id of the cell model for inference
        run_id: id of an existing run or of a new run. If none, will auto increment
        slide_id: id of the WSI. Only optional for cell inference.
        nuc_num_workers: number of workers for parallel processing of nuclei inference
        cell_num_workers: number of workers for parallel processing of cell inference
        score_threshold: nuclei network confidence cutoff for saving predictions
        max_detections: max nuclei detections for saving predictions
        nuc_batch_size: batch size for nuclei inference
        tile_size: size of the tile pushed to gpu for both nuclei and cell inference (Square tile, W = H)
        cell_batch_size: batch size for cell inference
        run_nuclei_pipeline: True if you want to perform nuclei detection
        run_cell_pipeline: True if you want to perform cell classification
        re_run_all_cell: True if you want to run all the cells instead of just remaining cells. With run_id and run_nuclei_pipeline = False to re run cell pipeline for all cells
        get_cuda_device_num: if you want the code to choose a gpu
        verbose: if you want to print the progress bar or not
    """
    device = get_device(get_cuda_device_num)

    # Create database connection
    db.init(db_name=db_name)

    if run_nuclei_pipeline:
        # Start timer for nuclei evaluation
        start = time.time()
        # Perform all nuclei evaluation
        run_id = nuclei_eval_pipeline(
            nuc_model_id,
            slide_id,
            run_id,
            tile_size,
            nuc_num_workers,
            nuc_batch_size,
            score_threshold,
            max_detections,
            device,
            verbose,
        )
        end = time.time()
        print(f"Nuclei evaluation time: {(end - start):.3f}")

    if run_cell_pipeline:
        # Start timer for cell evaluation
        start = time.time()
        # Perform all nuclei evaluation
        cell_eval_pipeline(
            project_name,
            organ_name,
            cell_model_id,
            run_id,
            cell_batch_size,
            cell_num_workers,
            tile_size,
            device,
            re_run_all_cell,
            verbose,
        )
        end = time.time()
        print(f"Cell evaluation time: {(end - start):.3f}")


def nuclei_eval_pipeline(
    model_id,
    slide_id,
    run_id,
    tile_size,
    num_workers,
    batch_size,
    score_threshold,
    max_detections,
    device,
    verbose,
):
    # Load model weights and push to device
    model = nuclei_infer.setup_model(model_id, device)
    # Load datasets and dataloader
    dataloader, pred_saver = nuclei_infer.setup_data(
        slide_id, run_id, model_id, tile_size, batch_size, overlap=200, num_workers=num_workers
    )
    # Predict nuclei
    nuclei_infer.run_nuclei_eval(
        dataloader, model, pred_saver, device, score_threshold, max_detections, verbose
    )
    nuclei_infer.clean_up(pred_saver)
    return pred_saver.id


def cell_eval_pipeline(
    project_name,
    organ_name,
    model_id,
    run_id,
    batch_size,
    num_workers,
    tile_size,
    device,
    re_run_all_cell,
    verbose,
):
    organ = get_organ(organ_name)
    # Load model weights and push to device
    model, model_architecture = cell_infer.setup_model(
        model_id, len(organ.cells), device
    )
    # Load datasets and dataloader
    dataloader, pred_saver = cell_infer.setup_data(
        run_id,
        model_id,
        model_architecture,
        tile_size,
        re_run_all_cell,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Setup or get path to embeddings hdf5 save location
    embeddings_path = cell_infer.setup_embedding_saving(project_name, pred_saver.id)
    # Predict cell classes
    cell_infer.run_cell_eval(
        dataloader, model, pred_saver, embeddings_path, device, verbose, re_run_all_cell
    )
    cell_infer.clean_up(pred_saver)


if __name__ == "__main__":

    logging.basicConfig(
        filename='app.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # 'w' overwrites, 'a' appends
    )


    try:
        typer.run(main)
    except Exception as err:
        logging.exception("An error occurred in main execution")


