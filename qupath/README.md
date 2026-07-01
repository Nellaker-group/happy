# Annotating ground truth in QuPath

How to create HAPPY training/ground-truth data in [QuPath](https://qupath.github.io/), and
how to load HAPPY predictions back in. The scripts here export QuPath annotations to the
CSV/TSV files the pipeline consumes.

| Script | Direction | Purpose |
|---|---|---|
| [`GetPointsInBox.groovy`](GetPointsInBox.groovy) | QuPath → CSV | Cell-class **points inside `TAnnot` boxes** → nuclei + cell training data |
| [`cellPointsToTissue.groovy`](cellPointsToTissue.groovy) | QuPath → CSV | Label points by the **tissue polygon** containing them → tissue training data |
| [`GetValTestRegions.groovy`](GetValTestRegions.groovy) | QuPath → CSV | Export `VAL_REGION`/`TEST_REGION` rectangles → optional spatial val/test splits |
| [`coord_to_tsv.py`](coord_to_tsv.py) | DB → QuPath | Export model predictions from the database to a TSV QuPath can load |

Every Groovy script has a `classNames` list and a `saveDir` at the top — **edit both** to
match your annotation classes (see `happy/organs.py`) and a valid absolute path on your
machine before running. The placenta class names are used as the example throughout;
rename them for other organs. **Save the QuPath project regularly while annotating.**

---

## 1. Create a project and load a WSI

1. Open QuPath → **Create Project** → choose/make an empty directory (e.g. `QuPath/`) → Open.
2. **Add Images**, drag your WSI into the window, set **Image type = Brightfield (H&E)** and
   **Rotation = No Rotation**.
3. Double-click the thumbnail to open the slide. Reopen later via **Open Project** →
   `project.qpproj`.

Note the slide's **pixel size** (µm/px) from the **Image** tab on the left — call it `S`.
You need it for the annotation box size below.

---

## 2. Nuclei detection + cell classification annotation

Both models train from the same point annotations: each nucleus is a point, its class is the
cell type. Workflow: **create a box → annotate every nucleus in it → repeat on diverse
regions → export**.

### Create the classes
Annotations tab → three dots (bottom right) → **Add/Remove → Add class**. Create:
- `TAnnot` (the annotation box class) and `Unknown`
- one class per cell type (placenta: `CYT, HOF, SYN, FIB, VEN, VMY, MAT, WBC, MES, EVT, KNT,
  EPI, MAC`), each with a distinct colour.

### Create an annotation box
**Objects → Specify annotation**, Type = **Rectangle**, Classification = **TAnnot**, sized to
match the model's training tiles (1600×1200 px at 0.1109 µm/px):

```
Width  = (0.1109 / S) * 1600
Height = (0.1109 / S) * 1200
```

(e.g. `S = 0.4942` → 359 × 269). The X/Y origin doesn't matter — click-drag to place the box
over a region with cells worth annotating. Build a **diverse** dataset across several boxes.

### Annotate inside the box
1. Open the **Points** tool (three circles). Click **Add** to make a point set, select the
   cell class in the Annotations pane and click **Set selected**. Make one set per cell class
   plus `Unknown`. Save the project, then reopen the Points tool.
2. In the **Counting** window pick a class, then click each nucleus in the box. Annotate
   **every** nucleus; if unsure of the class use `Unknown`.

### Export
**Automate → Show script editor**, paste [`GetPointsInBox.groovy`](GetPointsInBox.groovy),
edit `classNames` + `saveDir`, and **Run**. It writes `{slide}_from_groovy.csv`
(`bx,by,px,py,class`) — box origin plus point coords relative to the box.

### Build the training tiles + splits
```bash
python happy/microscopefile/make_tile_dataset.py --help
```
This crops tile images and writes the train/val/test annotation files for `nuc_train.py` and
`cell_train.py`.

---

## 3. Tissue classification annotation

Tissue labels come from cell points contained within hand-drawn tissue polygons.

1. **Create tissue classes** in the Annotations pane (placenta: `TVilli, SVilli, AVilli,
   MVilli, ImIVilli, MIVilli, Maternal, Chorion, Avascular, Fibrin, Sprout, Inflam`).
2. **Import the cell/nuclei prediction points** to label: Points tool → **Load** → the model
   prediction TSV (from [`coord_to_tsv.py`](coord_to_tsv.py); see §5).
3. **Draw polygon annotations** around each tissue region (any polygon tool), select the class
   and click **Set class**. Regions can be anywhere and need not connect.
4. Run [`cellPointsToTissue.groovy`](cellPointsToTissue.groovy) (edit `classNames` + `saveDir`).
   Each point is assigned the class of its containing polygon; points outside every polygon
   become `Unlabelled`. It writes `{slide}_tissue_points.csv` (`px,py,class`).

This CSV is the `--tissue-label-csv` input to `tissue_train.py`.

> **Multi-class / hierarchical tissue** (e.g. liver lobule steatosis + congestion) can be
> handled by running two adapted export scripts with a `prioritizedClassNames` list and class
> remapping. That is organ-specific — adapt `cellPointsToTissue.groovy` accordingly.

---

## 4. Validation and test regions (optional)

**By default you don't need to do this.** If you run `tissue_train.py` without
`--val-patch-files` / `--test-patch-files`, it splits nodes **randomly** (0.15 val / 0.15
test / 0.7 train; controlled by `include_validation`, default on). Provide patch files only
when you want a **spatially held-out** evaluation instead of a random split — this avoids
train/val spatial autocorrelation and gives a more rigorous test.

To make them, draw **rectangle** annotations in QuPath with classes **`VAL_REGION`** and
**`TEST_REGION`**, then run [`GetValTestRegions.groovy`](GetValTestRegions.groovy) (once with
`regionClass = "VAL_REGION"`, once with `"TEST_REGION"`) to export each rectangle's
`x,y,width,height` to a CSV (`{slide}_val_patches.csv` / `{slide}_test_patches.csv`) in the
project's `graph_splits/` dir, then pass them via `--val-patch-files` / `--test-patch-files`.

---

## 5. Loading model predictions into QuPath

To review/correct predictions, or to seed tissue annotation, export predictions from the
database and load them as points:

```bash
python qupath/coord_to_tsv.py \
    --organ-name placenta --project-name placenta \
    --run-id 3 --slide-name sample_wsi        # add --nuclei-only if no cell predictions
```

Then in QuPath:
1. Points tool → **Load points** → the `.tsv` (written to
   `projects/{project}/results/tsvs/{slide_name}.tsv`). Each cell class loads as its own
   Points object.
2. Ensure **Show annotations** is on; toggle **Fill annotations** and the **Point size** bar
   to taste. Untick **Highlight selected objects by colour** so points keep their class colour.
3. If the classes aren't present: Annotations tab → right-click → **Populate from existing
   objects → All classes**, then delete the QuPath built-in ones. Recolour classes (double-
   click) to a consistent scheme.
4. Use the **Move** tool to pan. You're now ready to correct predictions.

---

## 6. Exporting images (optional, for figures)

- **Whole WSI (no annotations):** File → Export → original pixels, PNG, downsample 20.
- **Whole WSI label mask:** use a `LabeledImageServer` script (`addLabel(class, value)` per
  class, `writeImage`) — order of labels matters.
- **Small section:** set magnification, toggle **Show annotations**, export snapshot as SVG.

---

## Notes / gaps

- The manual steps above are the intended workflow; confirm exact menu names against your
  QuPath version.
- Set each script's `saveDir` to a real path (the committed scripts use a placeholder
  `/../projects/...`).
- The VAL_REGION/TEST_REGION exporter (`GetValTestRegions.groovy`) is only needed for
  spatially held-out evaluation — a normal random-split train doesn't require it.
