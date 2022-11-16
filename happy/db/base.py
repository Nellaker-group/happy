from peewee import SqliteDatabase, Model

database = SqliteDatabase(None)


def init_db(db_name):
    # Local imports need to create all the tables and avoid circular import of BaseModel
    from happy.db.slides import Slide, Patient, Lab
    from happy.db.eval_runs import (
        EvalRun,
        Prediction,
        TileState,
        UnvalidatedPrediction,
    )
    from happy.db.models_training import Model, TrainRun
    from happy.db.tiles import (
        TrainTile,
        Annotation,
        Feature,
        TrainTileRun,
        TileFeature,
    )

    database.init(
        db_name,
        pragmas={
            "foreign_keys": 1,
            "journal_mode": "wal",
            "cache_size": 10000,
            "synchronous": 1,
        },
    )
    database.connect()
    database.create_tables(
        [
            Slide,
            Patient,
            Lab,
            EvalRun,
            Prediction,
            UnvalidatedPrediction,
            TileState,
            Model,
            TrainRun,
            TrainTile,
            Annotation,
            Feature,
            TrainTileRun,
            TileFeature,
        ]
    )


class BaseModel(Model):
    class Meta:
        database = database
