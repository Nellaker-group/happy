"""Bring any existing database up to the current schema without losing data.

Compares every registered model's fields against the live SQLite schema and:
  - Creates any tables that are entirely missing
  - Adds any columns that are present in the model but absent from the table

Safe to re-run: existing tables and columns are never modified or dropped.

Usage:
    python -m happy.db.migrate_db_schema
    python -m happy.db.migrate_db_schema --db-name /absolute/path/to/other.db
"""
from pathlib import Path

import typer
from peewee import ForeignKeyField, BooleanField

from happy.db.base import init_db, database
from happy.db.slides import Slide, Patient, Lab, ClinicalHistory
from happy.db.models_training import Model, TrainRun
from happy.db.graph_model import GraphModel
from happy.db.eval_runs import EvalRun, Prediction, TileState, UnvalidatedPrediction
from happy.db.tiles import TrainTile, Annotation, Feature, TrainTileRun, TileFeature

ALL_MODELS = [
    Lab, Patient, ClinicalHistory, Slide,
    TrainRun, Model,
    GraphModel,
    EvalRun, Prediction, UnvalidatedPrediction, TileState,
    TrainTile, Annotation, Feature, TrainTileRun, TileFeature,
]


def _column_def(field) -> str:
    """Return a minimal SQL column definition for ALTER TABLE ADD COLUMN."""
    sql_type = field.field_type  # e.g. 'INT', 'TEXT', 'REAL', 'DATETIME'

    parts = [sql_type]

    if isinstance(field, ForeignKeyField):
        ref_table = field.rel_model._meta.table_name
        parts.append(f"REFERENCES {ref_table}")
    elif not field.null and field.default is not None:
        default = field.default() if callable(field.default) else field.default
        if isinstance(field, BooleanField):
            parts.append(f"NOT NULL DEFAULT {1 if default else 0}")
        elif isinstance(default, str):
            parts.append(f"NOT NULL DEFAULT '{default}'")
        else:
            parts.append(f"NOT NULL DEFAULT {default}")

    return " ".join(parts)


def _existing_columns(table_name: str) -> set:
    rows = database.execute_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}  # row[1] is the column name


def migrate(db_path: Path):
    print(f"Migrating: {db_path}")
    init_db(db_path)

    # Pass 1: create any entirely missing tables (safe — skips existing ones)
    database.create_tables(ALL_MODELS, safe=True)

    # Pass 2: add any columns missing from existing tables.
    # Must happen before index creation so indexes are built on complete schema.
    total_added = 0
    for model in ALL_MODELS:
        table = model._meta.table_name
        existing = _existing_columns(table)

        for field_name, field in model._meta.fields.items():
            col = field.column_name  # handles ForeignKeyField (_id suffix)
            if col in existing:
                continue

            col_def = _column_def(field)
            try:
                database.execute_sql(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"
                )
                print(f"  ADDED   {table}.{col}  ({col_def})")
                total_added += 1
            except Exception as e:
                print(f"  ERROR   {table}.{col}: {e}")

    # Pass 3: rebuild all indexes so any that were created before their column
    # existed (ordering issue on first migration) are now consistent.
    if total_added > 0:
        database.execute_sql("REINDEX;")
        print(f"\nDone. Added {total_added} column(s) and rebuilt indexes.")
    else:
        print("Schema is already up to date — nothing to add.")


def main(db_name: str = "main.db"):
    if Path(db_name).is_absolute():
        db_path = Path(db_name)
    else:
        db_path = Path(__file__).parent.absolute() / db_name

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        raise typer.Exit(1)

    migrate(db_path)


if __name__ == "__main__":
    typer.run(main)
