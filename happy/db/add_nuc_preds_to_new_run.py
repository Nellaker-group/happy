import typer

import happy.db.eval_runs_interface as db


def main(
    run_id_to_copy: int = typer.Option(...),
):
    """Copy nuclei predictions from one evalrun to a new evalrun

    Args:
        run_id_to_copy: id of the evalrun to copy from
    """
    db.init()
    db.copy_nuclei_predictions(run_id_to_copy)


if __name__ == "__main__":
    typer.run(main)
