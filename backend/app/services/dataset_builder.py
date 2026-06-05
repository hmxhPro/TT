"""
app/services/dataset_builder.py
-------------------------------
Materialize a frozen YOLO dataset for a training job from a category's
uploaded images + their working annotations.

Layout produced (REQ2 — the annotated dataset is stored SEPARATELY from the
raw uploads, frozen per training job):

    datasets/<category_id>/raw/<image_id>.<ext>        # raw uploads (input)
    annotations/<category_id>/<image_id>.txt           # working labels (input)
    datasets/<category_id>/yolo/<job_id>/              # frozen output
        images/{train,val}/<image_id>.<ext>
        labels/{train,val}/<image_id>.txt
        dataset.yaml

Phase 1 is single-class per category: every box is class 0 and
`names: {0: <category name>}`.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Optional

import yaml  # provided by ultralytics' dependency set

from app.core.config import settings
from app.core.logging import logger


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _box_to_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Axis-aligned YOLO box (normalized cx,cy,w,h) → 4-point rectangle polygon
    [x1,y1, x2,y1, x2,y2, x1,y2], coords clamped to [0,1]."""
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    return [_clamp01(v) for v in (x1, y1, x2, y1, x2, y2, x1, y2)]


def _read_label(category_id: str, image_id: str) -> str:
    """Return YOLO **segmentation** label text for one image, class forced to 0
    (Phase 1 single-class).

    The configured base weights (yoloe-*-seg) are a segmentation model, so its
    training loss needs polygon masks — feeding it plain boxes makes the mask
    tensor empty and crashes F.interpolate. Each working annotation line is
    therefore normalized to a polygon:

      * box line     `c cx cy w h`          → axis-aligned rectangle polygon
      * polygon line `c x1 y1 x2 y2 ...`    → kept as-is (class → 0)

    The detection backend only ever reads boxes (image_detector reads r.boxes,
    never r.masks), so the rectangular masks are harmless at inference; this
    conversion exists purely to make seg fine-tuning of the base run.
    Missing/empty file → "" (a valid background sample)."""
    p = settings.ANNOTATIONS_DIR / category_id / f"{image_id}.txt"
    if not p.exists():
        return ""
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        coords = parts[1:]
        try:
            if len(coords) == 4:  # box → rectangle polygon
                cx, cy, w, h = (float(v) for v in coords)
                poly = _box_to_polygon(cx, cy, w, h)
            elif len(coords) >= 6 and len(coords) % 2 == 0:  # already a polygon
                poly = [_clamp01(float(v)) for v in coords]
            else:
                continue
        except ValueError:
            continue
        out.append("0 " + " ".join(f"{v:.6f}" for v in poly))
    return "\n".join(out)


def finalize(
    category_id: str,
    category_name: str,
    job_id: str,
    images: List[dict],
    val_split: Optional[float] = None,
    min_val: Optional[int] = None,
) -> dict:
    """Build the frozen YOLO dataset + dataset.yaml.

    `images` is a list of {"id", "stored_path"} for ANNOTATED images only.
    Returns {dataset_yaml, train_count, val_count, num_images, val_is_train}.
    """
    val_split = settings.TRAIN_VAL_SPLIT if val_split is None else val_split
    min_val = settings.MIN_VAL_IMAGES if min_val is None else min_val

    if not images:
        raise ValueError("没有已标注的图片，无法训练。请先标注至少一张图片。")

    root = settings.DATASETS_DIR / category_id / "yolo" / job_id
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    items = list(images)
    random.shuffle(items)
    n = len(items)
    n_train = max(1, int(round(n * val_split)))
    train_items = items[:n_train]
    val_items = items[n_train:]

    val_is_train = False
    if len(val_items) < min_val:
        # Too few images for a real holdout: train on all and mirror them into
        # val so Ultralytics can still compute mAP (metrics will be optimistic).
        train_items = items
        val_items = items
        val_is_train = True

    def _materialize(split: str, group: List[dict]) -> int:
        count = 0
        for it in group:
            src = Path(it["stored_path"])
            ext = src.suffix or ".jpg"
            dst_img = root / "images" / split / f"{it['id']}{ext}"
            try:
                shutil.copy2(src, dst_img)
            except FileNotFoundError:
                logger.warning(f"finalize: missing raw image {src}, skipping")
                continue
            (root / "labels" / split / f"{it['id']}.txt").write_text(
                _read_label(category_id, it["id"]), encoding="utf-8"
            )
            count += 1
        return count

    train_count = _materialize("train", train_items)
    val_count = _materialize("val", val_items)

    dataset_yaml = root / "dataset.yaml"
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: category_name},
    }
    dataset_yaml.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    logger.info(
        f"finalize: '{category_name}' job={job_id} train={train_count} "
        f"val={val_count} val_is_train={val_is_train} → {dataset_yaml}"
    )
    return {
        "dataset_yaml": str(dataset_yaml.resolve()),
        "train_count": train_count,
        "val_count": val_count,
        "num_images": n,
        "val_is_train": val_is_train,
    }
