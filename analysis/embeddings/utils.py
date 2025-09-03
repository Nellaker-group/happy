from pathlib import Path


def embeddings_results_path(embeddings_file, lab_id, slide_name, run_id):
    project_root = Path(str(embeddings_file).split("results")[0])
    vis_dir = (
        project_root
        / "visualisations"
        / "embeddings"
        / f"lab_{lab_id}"
        / f"slide_{slide_name}"
        / f"run_{run_id}"
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir


def setup(db, run_id):
    run = db.get_eval_run_by_id(run_id)
    slide_name = run.slide.slide_name
    lab_id = run.slide.lab
    print(f"Run id {run_id}, from lab {lab_id}, and slide {slide_name}")
    return lab_id, slide_name
