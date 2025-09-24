# Histology Analysis Pipeline.py (HAPPY) <img src="readme_images/HAPPYPlacenta.png" width="100" align="right" />

## [HAPPYv2.0 UPDATE Sept 2025](#update-to-happyv2)
HAPPY now has a new way to fetch images for nuclei and cell inferencing, and an improved graph model design for tissue inferencing. 

TLDR: previous graph models are no longer compatible due to architecture changes. Graph retraining or downloading new placenta tissue graph model weights.

## Overview

Accompanying repository for **HAPPY: A deep learning pipeline for mapping cell-to-tissue 
graphs across placenta histology whole slide images**. 

<img src="readme_images/Figure1.png" width="490" align="right" />

**Abstract**: _Accurate placenta pathology assessment is essential for managing maternal 
and newborn health, but the placenta's heterogeneity and temporal variability pose 
challenges for histology analysis. To address this issue, we developed the 
‘Histology Analysis Pipeline.PY’ (HAPPY), a deep learning hierarchical method for 
quantifying the variability of cells and micro-anatomical tissue structures across 
placenta histology whole slide images. HAPPY differs from patch-based features or 
segmentation approaches by following an interpretable biological hierarchy, representing 
cells and cellular communities within tissues at a single-cell resolution across whole 
slide images. We present a set of quantitative metrics from healthy term placentas as a 
baseline for future assessments of placenta health and we show how these metrics deviate 
in placentas with clinically significant placental infarction. HAPPY’s cell and tissue 
predictions closely replicate those from independent clinical experts and placental 
biology literature._

This repo contains all code for training, evaluating, and running inference across 
WSIs using the three stage deep learning pipeline detailed in the paper. The three 
deep learning steps are: **nuclei detection**, **cell classification** and **tissue 
classification**.

## Installation

Our codebase is writen in Python 3.10 and has been tested on Ubuntu 20.04.2 (WSL2), 
MacOS 15.1, and CentOS 7.9.2009 using both an NVIDIA A100 GPU and a CPU

You will first need to install the vips C binaries. The libvips documentation lists
installation instructions [here](https://github.com/libvips/libvips/wiki) for different 
OSs. If you are using MacOS you may brew install with:

```bash
brew install vips --with-cfitsio --with-imagemagick --with-openexr --with-openslide --with-webp
```

If you are on Ubuntu you may apt get:

```bash
sudo apt install libvips
```

For all remaining Python source code and dependencies, we recommend installation 
using the Makefile. Installation should only take a few minutes.

```bash
git clone git@github.com:Nellaker-group/happy.git
cd happy
# Activate venv environment with python installation:
# e.g. python3 -m venv happy-env
# source happy-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
# GPU (CUDA 12.1)
make environment_cu121
```
The make command will run the following:

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.4.0	
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r requirements.txt
pip install -e .
```
If you would rather install a different version of pytorch for your cuda version, 
please change the first two lines as per library instructions.
Makefile provides targets for CPU and different CUDA versions.

## Project Setup

The core code is organ-agnostic and may be used for any organ histology analysis. 
Organ-specific cell and tissue data may be added to `happy/organs.py`. We recommend
extending the core happy code by adding a new project to `projects/{project_name}`, 
using `projects/placenta` as a template.

If you would like to use the placenta histology training data and trained models from 
the paper, you may download the data from [this link](https://drive.google.com/drive/folders/1RvSQOxsWyUHf_SGV1Jzqa_Gc5QI4wQoy?usp=sharing). 
Keeping the same directory structure as in the link, place each directory into 
`projects/placenta`. This will allow you to train and evaluate all three models. For
a WSI inference demo, place the sample WSI section under 
`projects/placenta/slides/sample_wsi.tif`. We explain how to run the full 
inference pipeline using this slide as an example [here](#wsi-inference-pipeline).

## Training

This section provides a walkthrough on training the models for the placenta training data, but can be adapted for own dataset (using custom training data)[#making-custom-training-data].

### Nuclei Detection Training

Placenta nuclei detection training data from the paper should be placed under 
`projects/placenta/datasets/nuclei/` with annotations in 
`projects/placenta/annotations/nuclei/`. This data is split into respective data 
collection sources (i.e. 'hmc', 'uot', 'nuh') which are combined during training.

To train the nuclei detection model, run:

```bash
python nuc_train.py \
    --project-name placenta \
    --exp-name demo-train \
    --annot-dir annotations/nuclei \
    --dataset-names hmc --dataset-names uot --dataset-names nuh \
    --decay-gamma 0.5 \
    --frozen --init-from-inc
```

We recommend first fine tuning the model (pretrained on the coco dataset) using commands
`--frozen --init-from-inc`. Then loading the fine tuned model and training 
unfrozen using `--pre-trained {path} --no-frozen --no-init-from-inc`.

### Cell Classification Training

Placenta cell classification training data from the paper should be placed under 
`projects/placenta/datasets/cell_class/` with annotations in 
`projects/placenta/annotations/cell_class/`. This data is split into respective data 
collection sources (i.e. 'hmc', 'uot', 'nuh') which are combined during training.

To train the cell classification model, run:

```bash
python cell_train.py \
    --project-name placenta \
    --organ-name placenta \
    --exp-name demo-train \
    --annot-dir annotations/cell_class \
    --dataset-names hmc --dataset-names uot --dataset-names nuh \
    --decay-gamma 0.5 \
    --frozen --init-from-inc
```

As with the nuclei detection model, we recommend first fine tuning the model (pretrained 
on the imagenet dataset_ using commands `--frozen --init-from-inc`. Then loading the 
fine tuned model and training unfrozen using 
`--pre-trained {path} --no-frozen --no-init-from-inc`.

### Tissue Classification Training

By default, the training script will mask any nodes that are within the regions 
specified by validation and/or test .csv files within `graph_splits/` as validation
and/or test nodes. All other nodes will be marked as training nodes.  

We provide the training data and ground truth annotations for training the graph model 
across the cell graphs of two placenta WSIs, as per the paper. The training data should 
be placed under `projects/placenta/embeddings/` and the ground truth annotations in 
under `projects/placenta/annotations/graph/`.

To train the graph tissue model on this data, run:

```bash
python graph_train.py \
    --project-name placenta \
    --organ-name placenta \
    --exp-name demo_graph \
    --run-ids 1 --run-ids 2 \
    --tissue-label-tsvs wsi_1.tsv --tissue-label-tsvs wsi_2.tsv \
    --val-patch-files val_patches.csv \
    --test-patch-files test_patches.cs
```

This will train a supervised ClusterGCN (sup_clustergcn) tissue graph model.
NB: Model weights are saved under `projects/placenta/results/graph/sup_clustergcn/{exp_name}/{timestamp}/`
NB: Nuc and cell models later referenced via model ID from db, graph model weights referenced directly.

### Making Custom Training Data

We provide utility scripts for generating your own training data. 

**Nuclei Detection and Cell Classification:**
If you have used QuPath to create cell point annotations within boxes, you can:
1) Run the Groovy script `qupath/GetPointsInBox.groovy` to extract a .csv of these ground truth points and
classes.
2) From this, use `happy/microscopefile/make_tile_dataset.py` to
generate a dataset of tile images and train/val/test split annotation files from your
annotations for both nuclei detection and cell classification. 

**Tissue Classification:** In Qupath, if you load nuclei predictions onto your desired
WSI and draw polygon boundaries around different structures, you may use 
`qupath/cellPointsToTissues.groovy` to extract those points with ground truth tissue
labels.

## Evaluation

We provide evaluation scripts for checking model performance on validation or test 
data for each of the three models under `analysis/evaluation/`. The nuclei detection
model can be evaluated using `evaluate_nuclei_model.py`, the cell classification model 
can be evaluated using `evaluate_cell_model.py`, and the graph tissue model can be
evaluated using `evaluate_graph_model.py`.

## WSI Inference Pipeline

### Adding WSIs to the Database

You may add WSIs to the database using `happy/db/add_slides.py`. This will add all
slides with the specified file format at the specified directory to the database. We 
supply a starting database in github which contains two entries in the Slide and 
EvalRun tables to allow for training and evaluation of the graph model, as per the 
paper. 

### Adding Trained Models to the Database

You may add trained models to the database using `happy/db/add_model.py`. The sample 
starting database in github already contains data for both pretrained nuclei and 
cell models from the paper. They have model IDs 1 and 2 respectively.

NB: graph models are not currently used through the database, instead saved under
`projects/{project_name}/results/graph/{model_type}/{exp_name}/{timestamp}/` during training.
(Though can be added to database for completeness)

### Cell Pipeline

The cell pipeline `cell_inference.py` will run both nuclei detection and cell 
classification across a WSI. It will save each 'run' over a WSI into the Evalruns table 
in the database with respective predictions in the Predictions table. 
Each run can be stopped and restarted at any time. See the demo below for an example.

You may extract nuclei and cell predictions into a .tsv which QuPath can read using 
`qupath/coord_to_tsv.py`.

### Tissue Pipeline

Once you have nuclei and cell predictions, you may run the tissue pipeline 
`graph_inference.py`. This will construct a cell graph across the WSI and run the 
graph model. The pipeline will save a visualisation of tissue predictions and a .tsv 
file containing these predictions at the location of the trained model. See the demo
below for an example. 

### Demo Walkthrough

<img src="readme_images/demo_sample.png" width="300" align="right" />

1) Setup model weights
Save the nuclei and cell model weights under `projects/placenta/trained_models/`
These mdoels have the IDs in the database (e.g. nuc model = 1, cell model = 2)
Save the tissue model under `projects/placenta/results/graph/sup_clustergcn/demo_tissue/demo_timestamp`
   
2) Add the demo slide section at `projects/placenta/slides/sample_wsi.tif` to 
the database using:

```bash
CWD=$(pwd) # save absolute current working directory
python happy/db/add_slides.py \
    --slides-dir "$CWD/projects/placenta/slides/" \
    --lab-country na \
    --primary-contact na \
    --slide-file-format .tif \
    --pixel-size 0.2277
```

3) Run the nuclei and cell inference pipeline on this sample:

```bash
python cell_inference.py \
    --project-name placenta \
    --organ-name placenta \
    --nuc-model-id 1 \
    --cell-model-id 2 \
    --slide-id 3 \
    --cell-batch-size 100
```

4) Run the graph tissue inference pipeline on the nuclei and cell predictions:

```bash
python graph_inference.py \
    --project-name placenta \
    --organ-name placenta \
    --exp-name demo_tissue \
    --model-weights-dir demo_timestamp \
    --model-name graph_model.pt \
    --model-type sup_clustergcn \
    --run-id 3
```

At the location of the graph model weights, you will find an `eval/` directory which will
contain a visualisation of the tissue predictions and a .tsv file containing the 
predictions, which can be loaded into QuPath.

## Visualisation

Along with the visualisation generated by `graph_inference.py`, we also provide scripts
for visualising nuclei ground truth over training data in 
`analysis/evaluation/vis_groundtruth_nuclei.py` and nuclei predictions `analysis/evaluation/vis_nuclei_preds.py` 
and cell predictions in `vis_cell_preds.py`, regions of the cell
graph in `analysis/evaluation/vis_graph_patch.py`, and the ground truth tissue points 
in `analysis/evaluation/vis_groundtruth_graph.py`.


## Update to HAPPYv2

HAPPY v2.0 introduces updats to the pipeline, for speed and more robust workflows.

### Faster image fetching (nuc + cell pipeline)

<ins>Before</ins>: Readers (e.g.OpenSlide) fetches the huge number of 1600px by 1200px nuclei tile and 200px by 200 px cell tiles directly from the WSI to CPU, leading to high I/O restraint for slow inference speed.
        
<ins>After</ins>: Readers now fetched a large (default 15000px by 15000px) tile from the WSI to the CPU first, then crop the corresponding small tiles from the big tiles on the CPU memory before sending the small tiles to the GPU for inferencing.

Also integrates (histolab)[https://histolab.readthedocs.io/en/latest/tissue_masks.html] to first mask the image to avoid reading blank regions.


## Graph pipeline improvements

== graph models trained with the old architecture are no longer compatible, require using new model weights or retraining with updated pipeline ==

Updates: standardisation :Added `--standardise` flag to normalise embeddings (zero mean, unit variance) before training instead of using raw embeddings as node features. This gives an easy performance boost.
Improved the clusterGCN to mimic the cell model's final layers including adding batch normalisation.






