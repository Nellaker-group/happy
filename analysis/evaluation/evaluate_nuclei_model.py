"""Evaluate nuclei detection models — YOLO or RetinaNet, by box-mAP or point matching.
"""
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer
import yaml

# Make the repo root importable when run as a script 
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from happy.train.point_eval import evaluate_points_in_image


class ModelType(str, Enum):
    yolo = "yolo"
    retinanet = "retinanet"


class Metric(str, Enum):
    map = "map"
    point = "point"



def main(
    model_type: ModelType = typer.Option(..., "--model-type", help="Detector: yolo or retinanet"),
    metric: Metric = typer.Option(Metric.point, "--metric", help="Scoring: map (box mAP50, YOLO only) or point (30px F1)"),
    weights: List[str] = typer.Option(..., help="Path(s) to model weights. Pass multiple for several seeds/sizes."),
    data: List[str] = typer.Option(..., help="Path(s) to dataset YAML. Pass multiple to evaluate across organs."),
    output_dir: str = typer.Option(..., help="Directory to write the metrics CSV (+ plots for map)."),
    model_labels: Optional[List[str]] = typer.Option(None, help="Display name per weights (defaults to dir name)"),
    organ_labels: Optional[List[str]] = typer.Option(None, help="Display name per dataset (defaults to YAML stem)"),
    split: str = typer.Option("test", help="Dataset split: test or val"),
    # YOLO inference params (map + point)
    imgsz: int = typer.Option(1280, help="YOLO inference image size (matches production)"),
    conf: float = typer.Option(0.2, help="YOLO confidence threshold (matches production)"),
    iou: float = typer.Option(0.7, help="YOLO NMS IoU threshold (matches production)"),
    max_det: int = typer.Option(600, help="YOLO max detections per tile"),
    example_images: Optional[List[str]] = typer.Option(None, help="map only: tiles to overlay predicted boxes on"),
    # point matching
    valid_distance: float = typer.Option(30.0, help="point: match radius in px"),
    # RetinaNet inference params (point)
    score_threshold: float = typer.Option(0.5, help="retinanet: detection confidence threshold"),
    max_detections: int = typer.Option(500, help="retinanet: max detections per tile"),
    resnet_depth: int = typer.Option(101, help="retinanet: ResNet backbone depth of the checkpoint"),
    get_cuda_device_num: bool = typer.Option(False, help="retinanet: pass to happy get_device"),
):
    """Evaluate one or more nuclei models across one or more organ test sets.

    Single model + single dataset: a one-row CSV. Multiple models and/or datasets: a full
    generalisation (model × organ) matrix (and, for --metric map, heatmaps + bar charts).

        help usage:
    Choose the detector with
    ``--model-type`` and the scoring rule with ``--metric``:

    --model-type yolo      --metric map     box-IoU mAP50 via ultralytics val
    --model-type yolo      --metric point   YOLO box centres, 30 px point matching to find nuc (recommended)
    --model-type retinanet --metric point   RetinaNet box centres, 30 px point matching to find nuc (recommended)
             nb       RetinaNet has no box-mAP

    Pass multiple --weights and/or --data to build a full generalisation (model × organ) matrix.

    python analysis/evaluation/evaluate_nuclei_model.py --model-type yolo --metric point \
    --weights .../best.pt --data .../placenta_nuclei.yaml --output-dir <run_dir> \
    --model-labels yolo26s_s0 --organ-labels placenta --split test
    """
    if model_type is ModelType.retinanet and metric is Metric.map:
        typer.echo("ERROR: --model-type retinanet --metric map is unsupported (RetinaNet has no "
                   "box-mAP in this pipeline). Use --metric point for the YOLO-comparable F1.")
        raise typer.Exit(1)

    weight_paths = [Path(w).resolve() for w in weights]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model_labels and len(model_labels) != len(weight_paths):
        raise typer.BadParameter("--model-labels count must match --weights count.")
    if organ_labels and len(organ_labels) != len(data):
        raise typer.BadParameter("--organ-labels count must match --data count.")

    # Fallback display names mirror the original per-detector scripts: YOLO weights live at
    # <run>/yolo_output/weights/best.pt, RetinaNet at <run>/model_f1_*.pt.
    default_depth = 1 if model_type is ModelType.yolo else 0
    m_labels = model_labels or [p.parents[default_depth].name for p in weight_paths]
    o_labels = organ_labels or [Path(d).stem for d in data]

    if metric is Metric.map:
        _run_yolo_box(weight_paths, m_labels, data, o_labels, split, imgsz, conf, iou,
                      max_det, example_images, out)
        return

    if model_type is ModelType.yolo:
        predictor = lambda wp: _yolo_point_predictor(wp, imgsz, conf, iou, max_det)
    else:
        predictor = lambda wp: _retinanet_point_predictor(
            wp, score_threshold, max_detections, resnet_depth, get_cuda_device_num)
    _run_point(predictor, weight_paths, m_labels, data, o_labels, split, valid_distance, out)



# --- shared helpers --------------------------------------------------------
def _resolve_split_dirs(data_yaml: str, split: str):
    """Return (images_dir, labels_dir) for a split from an ultralytics dataset YAML."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", Path(data_yaml).parent)).resolve()
    images_dir = (root / cfg[split]).resolve()
    # YOLO convention: labels mirror images with .../images/... -> .../labels/...
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    return images_dir, labels_dir


def _load_gt_points(label_path: Path, width: int, height: int) -> np.ndarray:
    """Read a YOLO label file (cls cx cy w h, normalised) -> (N, 2) centre points in px."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.empty((0, 2))
    arr = np.loadtxt(label_path).reshape(-1, 5)
    xs = arr[:, 1] * width
    ys = arr[:, 2] * height
    return np.column_stack([xs, ys])


def _image_paths(images_dir: Path):
    return sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )


# --- predictors
def _yolo_point_predictor(weights: Path, imgsz: int, conf: float, iou: float, max_det: int):
    """Return predict(img_path) -> (centre_points Nx2, width, height) for a YOLO model."""
    from ultralytics import YOLO
    model = YOLO(str(weights))

    def predict(img_path: Path):
        r = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou,
                          max_det=max_det, verbose=False)[0]
        h, w = r.orig_shape
        pts = (r.boxes.xywh.cpu().numpy()[:, :2]
               if r.boxes is not None and len(r.boxes) > 0 else np.empty((0, 2)))
        return pts, w, h

    return predict


def _retinanet_point_predictor(weights: Path, score_threshold: float, max_detections: int,
                               resnet_depth: int, get_cuda_device_num: bool):
    """Return predict(img_path) -> (centre_points Nx2, width, height) for a RetinaNet model.

    Reuses the exact production preprocessing (happy.cell_infer.nuclei_infer): ImageNet
    Normalizer + Resizer, model(img, device), boxes /= scale, filter_by_score, boxes->points.
    """
    import torch
    from PIL import Image
    from happy.models import retinanet
    from happy.utils.utils import load_weights, get_device
    from happy.data.transforms.transforms import Normalizer, Resizer
    from happy.microscopefile.prediction_saver import PredictionSaver
    from happy.train.point_eval import convert_boxes_to_points

    device = get_device(get_cuda_device_num)
    model = retinanet.build_retina_net(num_classes=1, device=device, pretrained=False,
                                       resnet_depth=resnet_depth)
    model = load_weights(torch.load(str(weights), map_location=device), model)
    model = model.to(device)
    model.eval()
    normalizer, resizer = Normalizer(), Resizer()

    @torch.no_grad()
    def predict(img_path: Path):
        image = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        height, width = image.shape[:2]
        sample = resizer(normalizer({"img": image, "annot": np.zeros((0, 5), dtype=np.float32)}))
        resized, scale = sample["img"], sample["scale"]
        model_input = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        scores, _, boxes = model(model_input, device)
        boxes = boxes.cpu().numpy() / scale  # undo Resizer -> native-tile coordinates
        filtered = PredictionSaver.filter_by_score(max_detections, score_threshold,
                                                   scores.cpu().numpy(), boxes)
        pred_pts = convert_boxes_to_points(filtered)
        return (np.array(pred_pts) if len(pred_pts) else np.empty((0, 2))), width, height

    return predict


# --- point evaluation 
def _run_point(predictor, weight_paths, m_labels, data, o_labels, split, valid_distance, out):
    rows = []
    for m_label, wp in zip(m_labels, weight_paths):
        typer.echo(f"\nLoading model: {m_label}  ({wp})")
        predict = predictor(wp)
        for o_label, dp in zip(o_labels, data):
            images_dir, labels_dir = _resolve_split_dirs(dp, split)
            paths = _image_paths(images_dir)
            typer.echo(f"  Evaluating on {o_label} ({split}, {len(paths)} tiles, r={valid_distance:g}px)...")

            tp = fp = fn = 0
            for img_path in paths:
                pred_pts, w, h = predict(img_path)
                gt_pts = _load_gt_points(labels_dir / f"{img_path.stem}.txt", w, h)
                if len(gt_pts) == 0:
                    fp += len(pred_pts)
                    continue
                if len(pred_pts) == 0:
                    fn += len(gt_pts)
                    continue
                t, f, n = evaluate_points_in_image(gt_pts, pred_pts, valid_distance)
                tp += t; fp += f; fn += n

            precision = round(tp / (tp + fp), 3) if (tp + fp) else 0.0
            recall = round(tp / (tp + fn), 3) if (tp + fn) else 0.0
            f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0
            typer.echo(f"    F1={f1:.3f}  P={precision:.3f}  R={recall:.3f}  (TP={tp} FP={fp} FN={fn})")
            rows.append({"model": m_label, "organ": o_label, "f1": f1,
                         "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn})

    df = pd.DataFrame(rows)
    csv_path = out / "point_metrics.csv"
    df.to_csv(csv_path, index=False)
    typer.echo(f"\nPoint metrics saved to {csv_path}")
    typer.echo(df.to_string(index=False))


# --- YOLO box (mAP) evaluation 
def _run_yolo_box(weight_paths, m_labels, data, o_labels, split, imgsz, conf, iou,
                  max_det, example_images, out):
    from ultralytics import YOLO

    rows, models = [], []
    for m_label, wp in zip(m_labels, weight_paths):
        typer.echo(f"\nLoading model: {m_label}")
        model = YOLO(str(wp))
        models.append(model)
        for o_label, dp in zip(o_labels, data):
            typer.echo(f"  Evaluating on {o_label} ({split})...")
            try:
                results = model.val(data=str(Path(dp).resolve()), split=split, imgsz=imgsz,
                                    conf=conf, iou=iou, max_det=max_det, verbose=False,
                                    save_json=False, project=str(out), name="yolo_val", exist_ok=True)
                rows.append({"model": m_label, "organ": o_label,
                             "mAP50": float(results.box.map50), "mAP50_95": float(results.box.map),
                             "precision": float(results.box.mp), "recall": float(results.box.mr)})
                typer.echo(f"    mAP50={results.box.map50:.4f}  mAP50-95={results.box.map:.4f}  "
                           f"P={results.box.mp:.4f}  R={results.box.mr:.4f}")
            except Exception as e:
                typer.echo(f"    WARNING: evaluation failed — {e}")
                rows.append({"model": m_label, "organ": o_label, "mAP50": float("nan"),
                             "mAP50_95": float("nan"), "precision": float("nan"), "recall": float("nan")})

    df = pd.DataFrame(rows)
    df.to_csv(out / "generalisation_metrics.csv", index=False)
    typer.echo(f"\nMetrics saved to {out / 'generalisation_metrics.csv'}")

    _plot_generalisation_matrix(df, out)
    if len(weight_paths) > 1 or len(data) > 1:
        _plot_bar_comparison(df, out)
    if example_images:
        _plot_predictions(models, m_labels, example_images, out, conf, iou, imgsz, max_det)


def _plot_generalisation_matrix(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    for metric in ("mAP50", "precision", "recall"):
        pivot = df.pivot(index="model", columns="organ", values=metric)
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.8), max(3, len(pivot) * 1.2)), dpi=150)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", vmin=0, vmax=1, ax=ax,
                    cbar_kws={"label": metric}, linewidths=0.5)
        ax.set_title(f"Generalisation matrix — {metric}")
        ax.set_xlabel("Test organ"); ax.set_ylabel("Model")
        plt.tight_layout()
        plt.savefig(out / f"generalisation_{metric}.png")
        plt.close()
    typer.echo(f"Generalisation heatmaps saved to {out}/")


def _plot_bar_comparison(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    organs = df["organ"].unique().tolist()
    fig, axes = plt.subplots(1, len(organs), figsize=(5 * len(organs), 4), dpi=150, sharey=True)
    if len(organs) == 1:
        axes = [axes]
    for ax, organ in zip(axes, organs):
        sub = df[df["organ"] == organ]
        ax.bar(range(len(sub)), sub["mAP50"], tick_label=sub["model"].tolist())
        ax.set_title(organ); ax.set_ylabel("mAP50"); ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        sns.despine(ax=ax)
    plt.suptitle("mAP50 by model and organ", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "bar_comparison_map50.png", bbox_inches="tight")
    plt.close()
    typer.echo(f"Bar comparison saved to {out / 'bar_comparison_map50.png'}")


def _plot_predictions(models, model_labels, image_paths, out, conf, iou, imgsz, max_det):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import Image
    for img_path in image_paths:
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), dpi=150)
        if n == 1:
            axes = [axes]
        for ax, model, label in zip(axes, models, model_labels):
            results = model.predict(source=img_path, conf=conf, iou=iou, imgsz=imgsz,
                                    max_det=max_det, verbose=False)
            boxes = (results[0].boxes.xyxy.cpu().numpy()
                     if results[0].boxes is not None else np.empty((0, 4)))
            ax.imshow(img_arr)
            for x1, y1, x2, y2 in boxes:
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                linewidth=1, edgecolor="lime", facecolor="none"))
            ax.set_title(f"{label}  ({len(boxes)} detections)", fontsize=10)
            ax.axis("off")
        plt.suptitle(Path(img_path).name, fontsize=9, y=1.01)
        plt.tight_layout()
        plt.savefig(out / f"predictions_{Path(img_path).stem}.png", bbox_inches="tight")
        plt.close()
        typer.echo(f"Prediction plot saved to {out / f'predictions_{Path(img_path).stem}.png'}")

if __name__ == "__main__":
    typer.run(main)
