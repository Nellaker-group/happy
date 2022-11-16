from datetime import datetime

from peewee import (
    IntegerField,
    FloatField,
    ForeignKeyField,
    BooleanField,
    DateTimeField,
    CompositeKey,
    TextField,
)

from happy.db.base import BaseModel
from happy.db.slides import Slide
from happy.db.models_training import Model


class EvalRun(BaseModel):
    timestamp = DateTimeField(formats="%Y-%m-%d %H:%M", default=datetime.utcnow())
    nuc_model = ForeignKeyField(Model, backref="eval_runs")
    cell_model = ForeignKeyField(Model, backref="eval_runs", null=True)
    slide = ForeignKeyField(Slide, backref="eval_runs")
    tile_width = IntegerField(default=1600)
    tile_height = IntegerField(default=1200)
    pixel_size = FloatField(default=0.1109)
    overlap = IntegerField(default=200)
    subsect_x = IntegerField(null=True)
    subsect_y = IntegerField(null=True)
    subsect_w = IntegerField(null=True)
    subsect_h = IntegerField(null=True)
    embeddings_path = TextField(null=True)
    nucs_done = BooleanField(default=False)
    cells_done = BooleanField(default=False)


class Prediction(BaseModel):
    run = ForeignKeyField(EvalRun, backref="predictions")
    x = IntegerField()
    y = IntegerField()
    cell_class = IntegerField(null=True)

    class Meta:
        primary_key = CompositeKey("run", "x", "y")


class TileState(BaseModel):
    run = ForeignKeyField(EvalRun, backref="tile_states")
    tile_index = IntegerField()
    tile_x = IntegerField()
    tile_y = IntegerField()
    done = BooleanField(default=False)

    class Meta:
        primary_key = CompositeKey("run", "tile_index")


class UnvalidatedPrediction(BaseModel):
    run = ForeignKeyField(EvalRun, backref="unvalidated_predictions")
    x = IntegerField()
    y = IntegerField()
    is_valid = BooleanField(default=False)

    class Meta:
        primary_key = CompositeKey("run", "x", "y")
