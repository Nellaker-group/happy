from pathlib import Path
from collections import defaultdict

from peewee import Tuple, ValuesList, EnclosedNodeList, chunked

from happy.db.slides import Slide
from happy.db.eval_runs import EvalRun, Prediction, TileState, UnvalidatedPrediction
from happy.db.models_training import Model
from happy.db.base import database, init_db


def init():
    db_name = "main.db"
    db_path = Path(__file__).parent.absolute() / db_name
    init_db(db_path)


# returns the path to a slide
def get_slide_path_by_id(slide_id):
    slide = Slide.get_by_id(slide_id)
    return Path(slide.lab.slides_dir) / slide.slide_name

# returns a slide
def get_slide_by_id(slide_id):
    return Slide.get_by_id(slide_id)


# returns the path to model_weights
def get_model_weights_by_id(model_id):
    model = Model.get_by_id(model_id)
    return model.architecture, model.path


# returns an eval run
def get_eval_run_by_id(run_id):
    eval_run = EvalRun.get_by_id(run_id)
    return eval_run

# returns the path to the embeddings file for that run
def get_embeddings_path(run_id, embeddings_dir=None):
    eval_run = EvalRun.get_by_id(run_id)
    if not eval_run.embeddings_path:
        slide = eval_run.slide
        lab_id = slide.lab.id
        slide_name = slide.slide_name

        embeddings_path = Path(f"lab_{lab_id}") / f"slide_{slide_name}"
        (Path(embeddings_dir) / embeddings_path).mkdir(parents=True, exist_ok=True)
        path_with_file = embeddings_path / f"run_{run_id}.hdf5"

        eval_run.embeddings_path = path_with_file
        eval_run.save()
        return str(path_with_file)
    else:
        return eval_run.embeddings_path


# Updates temporary run tile state table with a new tiles run state
def save_new_tile_state(run_id, tile_xy_list):
    fields = [TileState.run, TileState.tile_index, TileState.tile_x, TileState.tile_y]

    xs = [x[0] for x in tile_xy_list]
    ys = [y[1] for y in tile_xy_list]

    data = [(run_id, i, xs[i], ys[i]) for i in range(len(tile_xy_list))]

    with database.atomic():
        for batch in chunked(data, 8000):
            TileState.insert_many(batch, fields=fields).execute()


# Returns False if state is None, otherwise True
def run_state_exists(run_id):
    state = TileState.get_or_none(TileState.run == run_id)
    return True if state else False


def get_run_state(run_id):
    tile_states = (
        TileState.select(TileState.tile_x, TileState.tile_y)
        .where(TileState.run == run_id)
        .tuples()
    )

    return tile_states


def get_remaining_tiles(run_id):
    with database.atomic():
        tile_coords = (
            TileState.select(TileState.tile_index, TileState.tile_x, TileState.tile_y)
            .where((TileState.run == run_id) & (TileState.done == False))
            .dicts()
        )
    return tile_coords


def get_remaining_cells(run_id):
    with database.atomic():
        cell_coords = (
            Prediction.select(Prediction.x, Prediction.y)
            .where((Prediction.run == run_id) & (Prediction.cell_class.is_null(True)))
            .dicts()
        )
    return cell_coords


def mark_finished_tiles(run_id, tile_indexes):
    with database.atomic():
        query = TileState.update(done=True).where(
            (TileState.run == run_id) & (TileState.tile_index << tile_indexes)
        )
        query.execute()


def mark_nuclei_as_done(run_id):
    eval_run = EvalRun.get_by_id(run_id)
    eval_run.nucs_done = True
    eval_run.save()


def mark_cells_as_done(run_id):
    eval_run = EvalRun.get_by_id(run_id)
    eval_run.cells_done = True
    eval_run.save()


def save_pred_workings(run_id, coords):
    data = [{"run": run_id, "x": coord[0], "y": coord[1]} for coord in coords]
    UnvalidatedPrediction.insert_many(data).on_conflict_ignore().execute()


def get_all_unvalidated_nuclei_preds(run_id):
    preds = (
        UnvalidatedPrediction.select(UnvalidatedPrediction.x, UnvalidatedPrediction.y)
        .where(UnvalidatedPrediction.run == run_id)
        .dicts()
    )
    # turns list of dicts into a dict of lists
    return {k: [dic[k] for dic in preds] for k in preds[0]}


def validate_pred_workings(run_id, valid_coords):
    print(f"marking {len(valid_coords)} nuclei as valid ")
    batch = 100000
    with database.atomic():
        for i in range(0, len(valid_coords), batch):
            coords_vl = ValuesList(valid_coords[i : i + batch], columns=("x", "y"))
            rhs = EnclosedNodeList([coords_vl])
            query = UnvalidatedPrediction.update(is_valid=True).where(
                (UnvalidatedPrediction.run == run_id)
                & (Tuple(UnvalidatedPrediction.x, UnvalidatedPrediction.y) << rhs)
            )
            query.execute()


def commit_pred_workings(run_id):
    source = (
        UnvalidatedPrediction.select(
            UnvalidatedPrediction.run, UnvalidatedPrediction.x, UnvalidatedPrediction.y
        )
        .where(
            (UnvalidatedPrediction.run == run_id)
            & (UnvalidatedPrediction.is_valid == True)
        )
        .order_by(UnvalidatedPrediction.x, UnvalidatedPrediction.y.asc())
    )

    rows = Prediction.insert_from(
        source, fields=[Prediction.run, Prediction.x, Prediction.y]
    ).execute()
    print(f"added {rows} nuclei predictions to Predictions table for eval run {run_id}")


def save_cells(run_id, coords, predictions):
    # split the coordinates by class prediction
    _dict = defaultdict(list)
    for i, pred in enumerate(predictions):
        _dict[pred].append(coords[i])

    # Update class prediction db in batches by each class
    with database.atomic():
        for pred in _dict.keys():
            coords_vl = ValuesList(_dict[pred], columns=("x", "y"))
            rhs = EnclosedNodeList([coords_vl])
            query = Prediction.update(cell_class=pred).where(
                (Prediction.run == run_id) & (Tuple(Prediction.x, Prediction.y) << rhs)
            )
            query.execute()


def get_num_remaining_tiles(run_id):
    return (
        TileState.select()
        .where((TileState.run == run_id) & (TileState.done == False))
        .count()
    )


def get_num_remaining_cells(run_id):
    return (
        Prediction.select()
        .where((Prediction.run == run_id) & (Prediction.cell_class.is_null(True)))
        .count()
    )


def get_total_num_nuclei(run_id):
    return Prediction.select().where(Prediction.run == run_id).count()


def get_predictions(run_id):
    return (
        Prediction.select(Prediction.x, Prediction.y, Prediction.cell_class)
        .where(Prediction.run == run_id)
        .dicts()
    )


def get_all_prediction_coordinates(run_id):
    return (
        Prediction.select(Prediction.x, Prediction.y)
        .where(Prediction.run == run_id)
        .dicts()
    )


def get_nuclei_in_range(run_id, min_x, min_y, max_x, max_y):
    # Get predictions within specified range
    return (
        Prediction.select(Prediction.x, Prediction.y, Prediction.cell_class)
        .where(
            (Prediction.run == run_id)
            & (Tuple(Prediction.x, Prediction.y) >= (min_x, min_y))
            & (Tuple(Prediction.x, Prediction.y) <= (max_x, max_y))
        )
        .tuples()
    )

def get_slide_pixel_size_by_evalrun(run_id):
    slide = Slide.select().join(EvalRun).where(EvalRun.id == run_id).get()
    return slide.pixel_size
