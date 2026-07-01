from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


_WEIGHTS_CACHE = Path.home() / ".cache" / "ultralytics"
# so that weights don't save to home dir each time


def _resolve_weights(model_name_or_path: str):
    p = Path(model_name_or_path)
    if p.is_absolute() or (p.parent != Path(".") and p.exists()):
        return str(p)
    if p.exists():
        return str(p.resolve())
    _WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    return str(_WEIGHTS_CACHE / p.name)


@dataclass
class YoloConfig:
    data: str
    model_name: str = "yolo26n.pt"
    epochs: int = 100
    imgsz: int = 1600
    batch: int = 16
    device: str = "cuda"
    workers: int = 4
    optimizer: str = "AdamW"
    lr0: float = 0.001
    weight_decay: float = 0.0005
    patience: int = 20
    project: str = "placenta"
    name: str = "yolo26n_train"
    pre_trained: Optional[str] = None
    single_cls: bool = False
    seed: int = 0
    # --- data augmentations ---
    degrees: float = 180.0  # full rotation: nuclei are rotation-invariant
    flipud: float = 0.5  # vertical flip
    fliplr: float = 0.5  # horizontal flip
    hsv_h: float = 0.02  # stain hue jitter
    hsv_s: float = 0.7  # stain saturation jitter
    hsv_v: float = 0.4  # brightness jitter
    scale: float = 0.5  # nucleus size variation (+/- 50%)
    copy_paste: float = 0.3  # paste nuclei into other tiles (dense-instance aug)
    copy_paste_mode: str = "flip"  # flips the paste nuclei to avoid duplicates
    close_mosaic: int = 40  # disable mosaic for the final N epochs
    max_det: int = 600  # max detections per tile (default YOLO is 300; dense tiles need more)
    # --- loss weights: bias training toward FINDING nuclei, not box tightness ---
    # ultralytics defaults are box=7.5, cls=0.5, dfl=1.5 (≈95% of loss is box geometry).
    box: float = 5.0  # down-weight box regression (was 7.5)
    cls: float = 1.0  # up-weight nuclei or background classification (was 0.5)
    dfl: float = 1.0  # down-weight box-edge refinement (was 1.5)
    # --- best.pt / early-stop selection metric (overrides ultralytics' default 100% mAP50-95) ---
    fitness_map50_w: float = 0.85  # weight on mAP@0.5 (detection at loose IoU)
    fitness_recall_w: float = 0.15  # weight on recall (finding nuclei)


def train_yolo(cfg: YoloConfig, run_path: Path) -> tuple[Path, Path]:
    """Train a YOLO model and return (best_weights, last_weights).

    Ultralytics output is written directly into run_path/yolo_output/.
    """
    # want nucleus centroids, not tight boxes, so select best.pt (and
    # early-stop) on detection quality instead of yolos' default fitness of
    # pure mAP@0.5:0.95 (box tightness). patch the box Metric.fitness
    # weights: [P, R, mAP@0.5, mAP@0.5:0.95].
    from ultralytics.utils.metrics import Metric as _UlMetric

    _fit_w = [0.0, cfg.fitness_recall_w, cfg.fitness_map50_w, 0.0]

    def _detection_fitness(self) -> float:
        return float((np.nan_to_num(np.array(self.mean_results())) * _fit_w).sum())

    _UlMetric.fitness = _detection_fitness

    weights = _resolve_weights(cfg.pre_trained if cfg.pre_trained else cfg.model_name)
    model = YOLO(weights)

    try:
        model.train(
            data=cfg.data,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=cfg.device,
            workers=cfg.workers,
            optimizer=cfg.optimizer,
            lr0=cfg.lr0,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            project=str(run_path),
            name="yolo_output",
            pretrained=True,
            verbose=True,
            exist_ok=True,
            single_cls=cfg.single_cls,
            seed=cfg.seed,
            degrees=cfg.degrees,
            flipud=cfg.flipud,
            fliplr=cfg.fliplr,
            hsv_h=cfg.hsv_h,
            hsv_s=cfg.hsv_s,
            hsv_v=cfg.hsv_v,
            scale=cfg.scale,
            copy_paste=cfg.copy_paste,
            copy_paste_mode=cfg.copy_paste_mode,
            close_mosaic=cfg.close_mosaic,
            max_det=cfg.max_det,
            box=cfg.box,
            cls=cfg.cls,
            dfl=cfg.dfl,
        )
    except Exception as e:
        print(f"Training interrupted: {e}")

    best_weights = run_path / "yolo_output" / "weights" / "best.pt"
    last_weights = run_path / "yolo_output" / "weights" / "last.pt"
    return best_weights, last_weights


def read_best_metrics(run_path: Path) -> tuple[float, int]:
    """Read best mAP@0.5 and number of epochs run from ultralytics' results.csv.

    Returns (best_map50, num_epochs); (0.0, 0) if results.csv is missing.
    """
    results_csv = run_path / "yolo_output" / "results.csv"
    if not results_csv.exists():
        return 0.0, 0
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    best_map50 = float(df["metrics/mAP50(B)"].max()) if "metrics/mAP50(B)" in df else 0.0
    return best_map50, len(df)


# ---------------------------------------------------------------------------
# Multi-dataset YAML merging
# ---------------------------------------------------------------------------

def merge_yamls(yaml_paths: List[str], run_path: Path) -> str:
    """Combine multiple single-organ YOLO YAMLs into one and save it to the run dir.

    Each YAML's train/val/test entries are resolved to absolute paths and collected
    into lists.
    """
    train_dirs, val_dirs, test_dirs = [], [], []
    names = None

    for yp in yaml_paths:
        with open(yp) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg.get("path", "")).resolve()

        def _abs(rel):
            p = Path(rel)
            return str(p if p.is_absolute() else base / p)

        for split, bucket in [("train", train_dirs), ("val", val_dirs), ("test", test_dirs)]:
            if split in cfg:
                entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
                bucket.extend(_abs(e) for e in entries)

        if names is None:
            names = cfg.get("names", [])
        elif cfg.get("names") != names:
            raise ValueError(
                f"Class names mismatch between YAMLs: {names} vs {cfg.get('names')} in {yp}"
            )

    combined = {"train": train_dirs, "val": val_dirs, "names": names}
    if test_dirs:
        combined["test"] = test_dirs

    out_path = run_path / "combined_dataset.yaml"
    with open(out_path, "w") as f:
        yaml.dump(combined, f, sort_keys=False)

    print(f"Combined YAML ({len(yaml_paths)} datasets) written to {out_path}")
    return str(out_path)
