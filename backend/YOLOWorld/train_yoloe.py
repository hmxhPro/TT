"""
YOLOE training CLI.
-------------------
Fine-tune / train a YOLOE (Ultralytics) model on a custom dataset.

Usage:
    python train_yoloe.py --data dataset.yaml [options]

The trained weights land at:
    <project>/<name>/weights/best.pt
Copy that absolute path into backend/.env's YOLO_WORLD_MODEL to swap the
detection backend over to the new weights.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Force offline mode for ultralytics (matches yolo_world_detector.py:55)
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")

# Make app.* importable so we can reuse logger / settings
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging import logger  # noqa: E402


def _load_yoloe_model(weights: str):
    """Instantiate a YOLOE model, falling back to YOLO if YOLOE is unavailable."""
    try:
        from ultralytics import YOLOE  # type: ignore

        logger.info(f"Loading YOLOE base weights: {weights}")
        return YOLOE(weights)
    except ImportError:
        logger.warning(
            "ultralytics.YOLOE not found — falling back to ultralytics.YOLO. "
            "Upgrade with: pip install -U 'ultralytics>=8.3.0' for full YOLOE support."
        )
        from ultralytics import YOLO

        logger.info(f"Loading YOLO base weights: {weights}")
        return YOLO(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train / fine-tune a YOLOE model on a custom dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML (see dataset.yaml.example).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.YOLO_WORLD_MODEL,
        help="Pretrained weights to start from (e.g. yoloe-11l-seg.pt). "
             "Defaults to YOLO_WORLD_MODEL in .env.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; pass -1 to let Ultralytics auto-fit GPU memory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=settings.DEVICE,
        help="Compute device, e.g. 'cuda:0', 'cpu', or '0,1' for multi-GPU.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(_BACKEND_ROOT / "runs" / "train"),
        help="Root output directory for training runs.",
    )
    parser.add_argument("--name", type=str, default="yoloe_exp", help="Experiment subdirectory name.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint in <project>/<name>.",
    )
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes.")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N layers (useful for light fine-tuning).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate (Ultralytics default if unset).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset YAML not found: {data_path}")
        return 1

    logger.info("─" * 60)
    logger.info("YOLOE training run")
    logger.info(f"  data    : {data_path}")
    logger.info(f"  model   : {args.model}")
    logger.info(f"  epochs  : {args.epochs}")
    logger.info(f"  imgsz   : {args.imgsz}")
    logger.info(f"  batch   : {args.batch}")
    logger.info(f"  device  : {args.device}")
    logger.info(f"  project : {args.project}")
    logger.info(f"  name    : {args.name}")
    logger.info("─" * 60)

    model = _load_yoloe_model(args.model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
        patience=args.patience,
    )
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0

    start = time.time()
    results = model.train(**train_kwargs)
    elapsed = time.time() - start

    # Resolve best.pt path. Ultralytics exposes save_dir on the results object.
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    logger.info("─" * 60)
    logger.info(f"Training finished in {elapsed / 60:.1f} min")
    logger.info(f"  best weights : {best_pt}")
    logger.info(f"  last weights : {last_pt}")
    logger.info(
        "To use these weights in the detection backend, set in backend/.env:\n"
        f"    YOLO_WORLD_MODEL={best_pt}"
    )
    logger.info("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
