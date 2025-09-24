from peewee import TextField, IntegerField, FloatField, ForeignKeyField, BooleanField

from happy.db.base import BaseModel


class ClinicalHistory(BaseModel):
    gestational_week = IntegerField(null=True)
    villi_maturity = IntegerField(null=True)
    c_section = BooleanField(null=True)
    twin = BooleanField(null=True)


class Patient(BaseModel):
    name = TextField(null=True)
    diagnosis = TextField(null=True)
    clinical_history = ForeignKeyField(
        ClinicalHistory, backref="clinical_history", null=True
    )


class Lab(BaseModel):
    country = TextField()
    primary_contact = TextField()
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
    slide_diagnosis = TextField(null=True)
