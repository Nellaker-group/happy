"""Create a base.db that keeps slide/lab/model metadata but drops all eval-run data.

Used by the multi-slide inference workflow (see the README): each SLURM task starts
from a copy of this base.db so that concurrent tasks never write to the same file,
avoiding SQLite locking. Run once, on an interactive node (not the head node):

    cp main.db base.db
    python -m happy.db.make_base_db
    sqlite3 base.db "vacuum;"

Peewee recreates the dropped tables automatically on the next connection.
"""
from happy.db.base import database as db
from happy.db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction
from happy.db.eval_runs_interface import init

init(db_name="base.db")
db.drop_tables([EvalRun, TileState, Prediction, UnvalidatedPrediction])
