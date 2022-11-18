from pathlib import Path

from happy.db.eval_runs import EvalRun
from happy.microscopefile.microscopefile import MicroscopeFile
from happy.microscopefile.reader import Reader


# returns an ms_file object with values from db if supplied with run_id
def get_msfile(
    slide_id=None,
    run_id=None,
    nuc_model_id=None,
    cell_model_id=None,
    tile_width=1600,
    tile_height=1200,
    pixel_size=0.1109,
    overlap=400,
    subsect_x=None,
    subsect_y=None,
    subsect_w=None,
    subsect_h=None,
):
    if not run_id:
        # Creates a new run with a new run_id
        run = EvalRun.create(
            nuc_model=nuc_model_id,
            cell_model=cell_model_id,
            slide=slide_id,
            tile_width=tile_width,
            tile_height=tile_height,
            pixel_size=pixel_size,
            overlap=overlap,
            subsect_x=subsect_x,
            subsect_y=subsect_y,
            subsect_w=subsect_w,
            subsect_h=subsect_h,
        )
        print(f"no run id given, making new run with id {run.id}")
    else:
        run = EvalRun.get_or_none(EvalRun.id == run_id)
        if not run:
            # Creates a new run with a new run_id (supplied run_id wasn't valid)
            run = EvalRun.create(
                nuc_model=nuc_model_id,
                cell_model=cell_model_id,
                slide=slide_id,
                tile_width=tile_width,
                tile_height=tile_height,
                pixel_size=pixel_size,
                overlap=overlap,
                subsect_x=subsect_x,
                subsect_y=subsect_y,
                subsect_w=subsect_w,
                subsect_h=subsect_h,
            )
            print(f"no run with id {run_id}, making new run with id {run.id}")
        else:
            # Uses run_id to get and continue existing run
            if cell_model_id and run.cell_model is None:
                run.cell_model = cell_model_id
                run.save()
            print("using existing microscopefile")

    return _init_msfile(run)


# returns an ms_file object from existing eval run
def get_msfile_by_run(run_id):
    run = EvalRun.get_by_id(run_id)
    return _init_msfile(run)


def _init_msfile(run):
    full_slide_path = str(Path(run.slide.lab.slides_dir) / run.slide.slide_name)
    reader = Reader.new(full_slide_path, run.slide.lvl_x)
    return MicroscopeFile(
        run.id,
        reader,
        run.tile_width,
        run.tile_height,
        run.pixel_size,
        run.slide.pixel_size,
        run.overlap,
        run.subsect_x,
        run.subsect_y,
        run.subsect_h,
        run.subsect_w,
        run.nucs_done,
        run.cells_done,
    )
