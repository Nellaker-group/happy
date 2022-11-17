import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from happy.data.transforms.collaters import collater
from happy.microscopefile import prediction_saver
from happy.data.datasets.ms_dataset import NucleiDataset
from happy.models import retinanet
from happy.utils.graceful_killer import GracefulKiller
from happy.utils.utils import load_weights
from happy.data.transforms.transforms import Normalizer, Resizer
from happy.db.msfile_interface import get_msfile
import happy.db.eval_runs_interface as db


# Load model weights and push to device
def setup_model(project_dir, model_id, device):
    model_architecture, model_weights_path = db.get_model_weights_by_id(model_id)
    print(f"model pre_trained path: {project_dir / model_weights_path}")
    if model_architecture == "retinanet":
        model = retinanet.build_retina_net(
            num_classes=1, device=device, pretrained=False, resnet_depth=101
        )
        state_dict = torch.load(project_dir / model_weights_path)
        # Removes the module string from the keys if it's there.
        model = load_weights(state_dict, model)
    else:
        raise ValueError(f"{model_architecture} not supported")

    model = model.to(device)
    print("Pushed model to device")
    return model


# Load datasets and dataloader
def setup_data(slide_id, run_id, model_id, batch_size, overlap, num_workers):
    ms_file = get_msfile(
        slide_id=slide_id, run_id=run_id, nuc_model_id=model_id, overlap=overlap
    )
    pred_saver = prediction_saver.PredictionSaver(ms_file)
    print("loading datasets")
    remaining_data = np.array(db.get_remaining_tiles(ms_file.id))
    curr_data_set = NucleiDataset(
        ms_file, remaining_data, transform=transforms.Compose([Normalizer(), Resizer()])
    )
    print("datasets loaded")
    print("creating dataloader")
    dataloader = DataLoader(
        curr_data_set,
        num_workers=num_workers,
        collate_fn=collater,
        batch_size=batch_size,
    )
    print("dataloader ready")
    return dataloader, pred_saver


# Predict nuclei loop
def run_nuclei_eval(
    dataset, model, pred_saver, device, score_threshold=0.3, max_detections=150
):
    # object for graceful shutdown. Current loop finishes on SIGINT or SIGTERM
    killer = GracefulKiller()
    early_break = False
    tiles_to_evaluate = db.get_num_remaining_tiles(pred_saver.id)
    model.eval()
    with torch.no_grad():
        with tqdm(total=tiles_to_evaluate) as pbar:
            for batch in dataset:
                if not killer.kill_now:
                    # find the indices in the batch which are and aren't empty tiles
                    empty_mask = np.array(batch["empty_tile"])
                    tile_indexes = np.array(batch["tile_index"])
                    empty_inds = tile_indexes[empty_mask]
                    non_empty_inds = tile_indexes[~empty_mask]

                    # if there are empty tiles in the batch, save them as empty
                    if empty_inds.size > 0:
                        for empty_ind in empty_inds:
                            pbar.update()
                            pred_saver.save_empty([empty_ind])

                    # if there are non-empty tiles in the batch,
                    # eval model and save predictions
                    if non_empty_inds.size > 0:
                        # filter out indices without images
                        non_empty_imgs = np.array(
                            batch["img"].cpu().numpy()[~empty_mask]
                        )
                        # Get scale factor
                        scale = np.array(batch["scale"])[~empty_mask][0]

                        # Network can't be fed batches of images
                        # as it returns predictions in one array
                        for i, non_empty_ind in enumerate(non_empty_inds):
                            # run network on non-empty images/tiles
                            model_input = torch.from_numpy(
                                np.expand_dims(non_empty_imgs[i], axis=0)
                            ).to(device)

                            scores, labels, boxes = model(model_input, device)
                            scores = scores.cpu().numpy()
                            boxes = boxes.cpu().numpy()

                            # Correct predictions from resizing of img.
                            boxes /= scale

                            # select indices which have a score above the threshold
                            image_boxes = pred_saver.filter_by_score(
                                max_detections, score_threshold, scores, boxes
                            )

                            pred_saver.save_nuclei(non_empty_ind, image_boxes)
                            pbar.update()
                else:
                    early_break = True
                    break

    if not early_break and not pred_saver.file.nucs_done:
        pred_saver.apply_nuclei_post_processing(cluster=True, remove_edges=True)
        pred_saver.commit_valid_nuclei_predictions()


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
