from datetime import datetime

from peewee import TextField, FloatField, DateTimeField

from happy.db.base import BaseModel


class GraphModel(BaseModel):
    path = TextField()
    hyperparameters_path = TextField(null=True)
    exp_name = TextField()
    model_type = TextField()
    organ = TextField()
    performance = FloatField(null=True)
    timestamp = DateTimeField(default=datetime.utcnow)
