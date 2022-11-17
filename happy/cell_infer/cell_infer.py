import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import h5py

from happy.db.msfile_interface import get_msfile
from happy.microscopefile.prediction_saver import PredictionSaver
from happy.data.datasets.ms_dataset import CellDataset
from happy.data.transforms.collaters import cell_collater
from happy.data.transforms.transforms import Normalizer, Resizer
from happy.utils.graceful_killer import GracefulKiller
from happy.models import resnet
from happy.utils.hdf5 import get_embeddings_file
import happy.db.eval_runs_interface as db


# Load model weights and push to device
def setup_model(project_dir, model_id, out_features, device):
    torch_home = Path(__file__).parent.parent.parent.absolute()
    os.environ["TORCH_HOME"] = str(torch_home)

    _, model_weights_path = db.get_model_weights_by_id(model_id)
    print(f"model pre_trained path: {project_dir / model_weights_path}")
    model = resnet.build_resnet(out_features=out_features, depth=50)
    model.load_state_dict(
        torch.load(project_dir / model_weights_path, map_location=device), strict=True
    )

    model = model.to(device)
    print("Pushed model to device")
    return model


# Load datasets and dataloader
def setup_data(run_id, model_id, batch_size, num_workers):
    ms_file = get_msfile(run_id=run_id, cell_model_id=model_id)
    pred_saver = PredictionSaver(ms_file)
    print("loading datasets")
    image_size = (224, 224)
    remaining_data = np.array(db.get_remaining_cells(ms_file.id))
    dataset = CellDataset(
        ms_file,
        remaining_data,
        transform=transforms.Compose(
            [
                Normalizer(),
                Resizer(
                    min_side=image_size[0],
                    max_side=image_size[1],
                    padding=False,
                    scale_annotations=False,
                ),
            ]
        ),
    )
    print("datasets loaded")
    print("creating dataloader")
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=cell_collater,
        batch_size=batch_size,
    )
    print("dataloader ready")
    return dataloader, pred_saver


# Setup or get path to embeddings hdf5 save location
def setup_embedding_saving(project_dir, run_id):
    embeddings_path = get_embeddings_file(project_dir, run_id)
    if not os.path.isfile(embeddings_path):
        total_cells = db.get_total_num_nuclei(run_id)
        with h5py.File(embeddings_path, "w-") as f:
            f.create_dataset("predictions", (total_cells,), dtype="int8")
            f.create_dataset("embeddings", (total_cells, 64), dtype="float32")
            f.create_dataset("confidence", (total_cells,), dtype="float16")
            f.create_dataset("coords", (total_cells, 2), dtype="uint32")
    return embeddings_path


# Predict cell classes loop
def run_cell_eval(dataset, cell_model, pred_saver, embeddings_path, device):
    # object for graceful shutdown. Current loop finishes on SIGINT or SIGTERM
    killer = GracefulKiller()
    early_break = False
    remaining = db.get_num_remaining_cells(pred_saver.id)
    cell_model.eval()

    def copy_data(module, input, output):
        embedding.copy_(output.data)

    with torch.no_grad():
        with tqdm(total=remaining) as pbar:
            for i, batch in enumerate(dataset):
                if not killer.kill_now:
                    # evaluate model and set up saving the embeddings layer
                    embedding = torch.zeros((batch["img"].shape[0], 64), device=device)
                    handle = cell_model.fc.embeddings_layer.register_forward_hook(
                        copy_data
                    )

                    # Calls forward() and copies the embedding data
                    class_prediction = cell_model(batch["img"].to(device).float())
                    # Removes the hook before the next forward() call
                    handle.remove()

                    # get predictions, confidence, and embeddings
                    _, predicted = torch.max(class_prediction.data, 1)
                    confidence = softmax(class_prediction.data, dim=1).cpu().numpy()
                    predicted = predicted.cpu().tolist()
                    embeddings = embedding.cpu().tolist()
                    top_confidence = confidence[(range(len(predicted)), [predicted])][0]

                    # setup values for saving in hdf5 file
                    num_to_save = len(predicted)
                    start = -remaining
                    end = -remaining + num_to_save

                    # save embeddings layer for each prediction in the batch
                    with h5py.File(embeddings_path, "r+") as f:
                        if end == 0:
                            end = len(f["predictions"])
                        f["predictions"][start:end] = predicted
                        f["embeddings"][start:end] = embeddings
                        f["confidence"][start:end] = top_confidence
                        f["coords"][start:end] = batch["coord"]
                    remaining -= num_to_save

                    # save the class predictions of the batch
                    pred_saver.save_cells(batch["coord"], predicted)
                    pbar.update(dataset.batch_size)
                else:
                    early_break = True
                    break

    if not early_break:
        pred_saver.finished_cells()


def clean_up(pred_saver):
    ms_file = pred_saver.file
    try:
        if isinstance(ms_file.reader, ms_file.reader.BioFormatsFile):
            print("shutting down BioFormats vm")
            ms_file.reader.stop_vm()
        else:
            pass
    except AttributeError:
        pass
