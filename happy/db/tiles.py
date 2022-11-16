from peewee import (
    TextField,
    IntegerField,
    FloatField,
    ForeignKeyField,
    BooleanField,
    DateTimeField,
    CompositeKey,
)

from happy.db.base import BaseModel
from happy.db.slides import Slide
from happy.db.models_training import TrainRun


class TrainTile(BaseModel):
    img_file_name = TextField()
    split_type = TextField()
    dataset_type = TextField()
    timestamp = DateTimeField(formats="%Y-%m-%d %H:%M:%S")
    slide = ForeignKeyField(Slide, backref="train_tiles")
    annotator = TextField()


class Annotation(BaseModel):
    tile = ForeignKeyField(TrainTile, backref="annotations")
    x = FloatField(null=True)
    y = FloatField(null=True)
    cell_class = IntegerField(null=True)

    class Meta:
        primary_key = CompositeKey("tile", "x", "y", "cell_class")


class Feature(BaseModel):
    tissue_type = TextField(null=True)
    healthy = BooleanField(null=True)


class TrainTileRun(BaseModel):
    train_run = ForeignKeyField(TrainRun)
    train_tile = ForeignKeyField(TrainTile)

    class Meta:
        primary_key = CompositeKey("train_run", "train_tile")


class TileFeature(BaseModel):
    train_tile = ForeignKeyField(TrainTile)
    feature = ForeignKeyField(Feature)

    class Meta:
        primary_key = CompositeKey("train_tile", "feature")
