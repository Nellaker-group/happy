-- Merge one task-specific db into main.db (multi-slide inference workflow).
-- Run from inside the task db:  sqlite3 task_dbs/1.db < merge_db.sql
-- Each task's run_ids are offset by main.db's current max evalrun id so they do not collide.

-- attach the main db to merge into the task one you opened with sqlite3
attach 'main.db' as main_db;

-- increment prediction.run_id from task db by max(id) and append to main.db
with max_id as (select max(id) as max_id from main_db.evalrun),
results as (
select run_id + (select max_id from max_id) as run_id, x, y, cell_class
from prediction
)
insert into main_db.prediction (run_id, x, y, cell_class) select * from results;

-- increment tilestate.run_id from task db by max(id) and append to main.db
with max_id as (select max(id) as max_id from main_db.evalrun),
results as (
select run_id + (select max_id from max_id) as run_id, tile_index, tile_x, tile_y, done
from tilestate
)
insert into main_db.tilestate (run_id, tile_index, tile_x, tile_y, done) select * from results;

-- increment unvalidatedprediction.run_id from task db by max(id) and append to main.db
with max_id as (select max(id) as max_id from main_db.evalrun),
results as (
select run_id + (select max_id from max_id) as run_id, x, y, is_valid
from unvalidatedprediction
)
insert into main_db.unvalidatedprediction (run_id, x, y, is_valid) select * from results;

-- NB: THIS STEP MUST GO LAST OR max(id) CHANGES
-- increment evalrun.id from task db by max(id) and append to main.db
with max_id as (select max(id) as max_id from main_db.evalrun),
results as (
select id + (select max_id from max_id) as id, timestamp, nuc_model_id, cell_model_id, slide_id, tile_width, tile_height, pixel_size, overlap, subsect_x, subsect_y, subsect_w, subsect_h, embeddings_path, nucs_done, cells_done
from evalrun
)
insert into main_db.evalrun (id, timestamp, nuc_model_id, cell_model_id, slide_id, tile_width, tile_height, pixel_size, overlap, subsect_x, subsect_y, subsect_w, subsect_h, embeddings_path, nucs_done, cells_done) select * from results;
