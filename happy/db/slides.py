from peewee import TextField, IntegerField, FloatField, ForeignKeyField, BooleanField

from happy.db.base import BaseModel


class Patient(BaseModel):
    diagnosis = TextField(null=True)
    clinical_history = TextField(null=True)


class Lab(BaseModel):
    country = TextField()
    primary_contact = TextField(null=True)
    slides_dir = TextField()
    study_name = TextField(null=True)
    has_pathologists_notes = BooleanField(null=True)
    has_clinical_data = BooleanField(null=True)


class Slide(BaseModel):
    slide_name = TextField(unique=True)
    tissue_type = TextField(null=True)
    lvl_x = IntegerField(default=0)
    pixel_size = FloatField(null=True)
    lab = ForeignKeyField(Lab, backref="slides")
    patient = ForeignKeyField(Patient, backref="slides", null=True)
