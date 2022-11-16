from datetime import datetime

from peewee import TextField, IntegerField, FloatField, ForeignKeyField, DateTimeField

from happy.db.base import BaseModel


class TrainRun(BaseModel):
    run_name = TextField()
    timestamp = DateTimeField(formats="%Y-%m-%d %H:%M", default=datetime.utcnow())
    type = TextField()
    pre_trained_path = TextField(null=True)
    num_epochs = IntegerField()
    batch_size = IntegerField()
    init_lr = FloatField()
    lr_step = IntegerField(null=True)


class Model(BaseModel):
    train_run = ForeignKeyField(TrainRun, backref="models")
    type = TextField()
    path = TextField()
    architecture = TextField()
    performance = FloatField()
