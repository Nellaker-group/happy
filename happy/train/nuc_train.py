import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from happy.logger.benchmark_logger import BenchmarkLogger


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


def train_yolo(cfg: YoloConfig, run_path: Path) -> tuple[Path, Path, BenchmarkLogger]:
    """Train a YOLO model and return (best_weights, last_weights, BenchmarkLogger).

    Ultralytics output is written directly into run_path/yolo_output/.
    Per-epoch time and GPU memory are captured via callbacks.
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

    epoch_times: list[float] = []
    epoch_memory_mb: list[float] = []
    _epoch_start: list[float] = []

    def on_train_start(trainer):
        _save_architecture(trainer.model, run_path)

    def on_train_epoch_start(trainer):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _epoch_start.append(time.monotonic())

    def on_train_epoch_end(trainer):
        epoch_times.append(time.monotonic() - _epoch_start[-1])
        epoch_memory_mb.append(
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else float("nan")
        )

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

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
        print("Attempting to salvage partial metrics from results.csv...")

    best_weights = run_path / "yolo_output" / "weights" / "best.pt"
    last_weights = run_path / "yolo_output" / "weights" / "last.pt"

    bench = _build_benchmark_logger(model, epoch_times, epoch_memory_mb)
    return best_weights, last_weights, bench


def _save_architecture(model, run_path: Path):
    try:
        info = model.info(verbose=False) if hasattr(model, "info") else None
        if info is None:
            return
        layers, params, _, gflops = info
        pd.DataFrame([{
            "layers": int(layers),
            "parameters": int(params),
            "gflops": round(float(gflops), 1),
        }]).to_csv(run_path / "architecture.csv", index=False)
    except Exception as e:
        print(f"WARNING: could not save architecture info — {e}")


def _build_benchmark_logger(
    model, epoch_times: list[float], epoch_memory_mb: list[float]
):
    bench = BenchmarkLogger()

    results_csv = Path(model.trainer.save_dir) / "results.csv"
    if not results_csv.exists():
        return bench

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    try:
        train_size = len(model.trainer.train_loader.dataset)
    except Exception:
        train_size = None

    col_map = {
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
    }

    for i, (_, row) in enumerate(df.iterrows()):
        epoch = int(row.get("epoch", 0)) + 1
        epoch_t = epoch_times[i] if i < len(epoch_times) else float("nan")
        gpu_mb = epoch_memory_mb[i] if i < len(epoch_memory_mb) else float("nan")
        metrics = {"epoch_time_s": epoch_t, "gpu_memory_mb": gpu_mb}
        if train_size is not None and not (epoch_t != epoch_t):  # not nan
            metrics["imgs_per_sec"] = train_size / epoch_t
        for src, dst in col_map.items():
            if src in row:
                metrics[dst] = float(row[src])
        bench.log_epoch(epoch, metrics)

    return bench


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


# ---------------------------------------------------------------------------
# Stitching benchmark metrics across resumed runs
# ---------------------------------------------------------------------------

_YOLO_COL_MAP = {
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


def load_prev_benchmark(run_dir: Path):
    """Load benchmark metrics from a previous run.

    Prefers benchmark_metrics.csv; falls back to yolo_output/results.csv if the
    run timed out before our benchmark code could write (timing/memory will be NaN).
    """
    bench_csv = run_dir / "benchmark_metrics.csv"
    if bench_csv.exists():
        return pd.read_csv(bench_csv)

    yolo_csv = run_dir / "yolo_output" / "results.csv"
    if not yolo_csv.exists():
        return None

    print(f"WARNING: benchmark_metrics.csv not found in {run_dir} — falling back to "
          f"yolo_output/results.csv (epoch_time_s / gpu_memory_mb will be NaN).")
    raw = pd.read_csv(yolo_csv)
    raw.columns = raw.columns.str.strip()
    rows = []
    for _, row in raw.iterrows():
        r = {"epoch": int(row.get("epoch", 0)) + 1,
             "epoch_time_s": float("nan"),
             "gpu_memory_mb": float("nan"),
             "imgs_per_sec": float("nan")}
        for src, dst in _YOLO_COL_MAP.items():
            if src in row:
                r[dst] = float(row[src])
        rows.append(r)
    return pd.DataFrame(rows)


def stitch_benchmarks(previous_run_dir: str, current_bench: BenchmarkLogger) -> BenchmarkLogger:
    """Prepend a previous run's benchmark metrics to the current run's logger.

    Epoch numbers in the current run are offset so they continue
    from where the previous run left off.
    """
    prev_df = load_prev_benchmark(Path(previous_run_dir))
    if prev_df is None:
        print(f"WARNING: No benchmark data found in {previous_run_dir} — benchmark will cover this run only.")
        return current_bench
    epoch_offset = int(prev_df["epoch"].max())

    current_df = current_bench.to_df()
    current_df["epoch"] = current_df["epoch"] + epoch_offset

    combined_df = pd.concat([prev_df, current_df], ignore_index=True)

    stitched = BenchmarkLogger()
    for _, row in combined_df.iterrows():
        stitched.log_epoch(int(row["epoch"]), row.to_dict())
    return stitched
