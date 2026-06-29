import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from happy.data.transforms.collaters import collater
from happy.microscopefile import prediction_saver
from happy.data.datasets.ms_dataset import NucleiDataset
from happy.models import retinanet
from happy.utils.utils import load_weights, GracefulKiller
from happy.data.transforms.transforms import Normalizer, Resizer
from happy.db.msfile_interface import get_msfile
import happy.db.eval_runs_interface as db

from ultralytics import YOLO


# Load model weights and push to device
def setup_model(model_id, device):
    model_architecture, model_weights_path = db.get_model_weights_by_id(model_id)
    print(f"model pre_trained path: {model_weights_path}")
    if model_architecture == "retinanet":
        model = retinanet.build_retina_net(
            num_classes=1, device=device, pretrained=False, resnet_depth=101
        )
        state_dict = torch.load(model_weights_path)
        # Removes the module string from the keys if it's there.
        model = load_weights(state_dict, model)
        model = model.to(device)
    elif model_architecture == 'yolo26':
        model = YOLO(model_weights_path)  # use yolo to load the model weight directly
    else:
        raise ValueError(f"{model_architecture} not supported")

    print(f"Pushed model {model_architecture} to device")
    return model, model_architecture


# Load datasets and dataloader
def setup_data(slide_id, run_id, model_id, big_tile_size, batch_size, overlap, num_workers, model_architecture="retinanet"):
    ms_file = get_msfile(
        slide_id=slide_id, run_id=run_id, nuc_model_id=model_id, overlap=overlap
    )
    pred_saver = prediction_saver.PredictionSaver(ms_file)
    print("loading datasets")
    remaining_data = np.array(db.get_remaining_tiles(ms_file.id))
    # RetinaNet requires ImageNet normalisation (pretrained ResNet backbone) and
    # the Resizer to match its training resolution.
    if model_architecture == "retinanet":
        transform = transforms.Compose([Normalizer(), Resizer()])
    else:
        transform = None
    curr_data_set = NucleiDataset(
        ms_file, remaining_data, big_tile_size, transform=transform
    )
    print("datasets loaded")
    print("creating dataloader")
    dataloader = DataLoader(
        curr_data_set,
        num_workers=num_workers,
        collate_fn=collater,
        batch_size=batch_size,
        prefetch_factor=2,
        pin_memory=True
    )
    print("dataloader ready")
    return dataloader, pred_saver


# Predict nuclei
def run_nuclei_eval(
    dataset, model, model_architecture, pred_saver, device, score_threshold, max_detections, verbose
):
    if model_architecture == "retinanet":
        _run_retinanet_eval(dataset, model, pred_saver, device, score_threshold, max_detections, verbose)
    elif model_architecture == "yolo26":
        _run_yolo_eval(dataset, model, pred_saver, device, score_threshold, max_detections, verbose)
    else:
        raise ValueError(f"Unknown architecture for inference: {model_architecture}")


def _run_retinanet_eval(
    dataset, model, pred_saver, device, score_threshold, max_detections, verbose
):
    # object for graceful shutdown. Current loop finishes on SIGINT or SIGTERM
    killer = GracefulKiller()
    early_break = False
    tiles_to_evaluate = db.get_num_remaining_tiles(pred_saver.id)
    model.eval()
    with torch.no_grad():
        with tqdm(total=tiles_to_evaluate, disable=not verbose) as pbar:

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

                            scores, _, boxes = model(model_input, device)
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
                torch.cuda.empty_cache()

    if not early_break and not pred_saver.file.nucs_done:
        pred_saver.apply_nuclei_post_processing(cluster=True, remove_edges=True)
        pred_saver.commit_valid_nuclei_predictions()


def _run_yolo_eval(
    dataset, model, pred_saver, device, score_threshold, max_detections, verbose
):
    # object for graceful shutdown. Current loop finishes on SIGINT or SIGTERM
    killer = GracefulKiller()
    early_break = False
    tiles_to_evaluate = db.get_num_remaining_tiles(pred_saver.id)

    print("Running YOLO nuclei inference...")

    # Run inference at the same imgsz the model was trained on, read from checkpoint
    # if not, fall back to running on 1280 (resizes to this)
    try:
        train_imgsz = int(model.ckpt["train_args"]["imgsz"])
    except Exception:
        train_imgsz = 1280
    print(f"YOLO inference imgsz={train_imgsz}")

    # YOLO manages its own inference mode internally — no model.eval() / torch.no_grad() needed
    with tqdm(total=tiles_to_evaluate, disable=not verbose) as pbar:

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

                    keep = ~torch.as_tensor(empty_mask)  # tiles that actually have an image

                    imgs_list = [
                        np.ascontiguousarray(
                            (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
                        )
                        for t in batch["img"][keep]
                    ]

                    # Single batched forward pass for all non-empty tiles
                    results = model.predict(
                        imgs_list, imgsz=train_imgsz, conf=score_threshold,
                        iou=0.7, max_det=max_detections, device=device, verbose=False
                    )

                    # per-tile offset math + DB writes
                    for non_empty_ind, res in zip(non_empty_inds, results):
                        r = res.boxes  # this is a Boxes object

                        if r is not None and len(r) > 0:
                            boxes = r.xyxy.cpu().numpy()   # native 1600x1200 tile coords
                            scores = r.conf.cpu().numpy()  # shape: (N,)
                        else:
                            boxes = np.empty((0, 4))
                            scores = np.empty((0,))

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
