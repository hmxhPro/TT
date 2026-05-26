"""
YOLOWorld detector with SAHI integration.
-----------------------------------------
Provides YOLO-World v11 detection with SAHI slicing for small object detection.

YOLO-World is an open-vocabulary object detector that can detect objects
based on text prompts without fine-tuning. SAHI (Slicing Aided Hyper Inference)
improves detection of small objects by slicing large images into smaller patches.

Installation:
    pip install ultralytics>=8.3.0 sahi>=0.11.18

Model variants:
    - yolo11l-world (large, best accuracy)
    - yolo11m-world (medium, balanced)
    - yolo11s-world (small, fastest)
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.services.detector import BaseDetector, RawDetection


class YOLOWorldDetector(BaseDetector):
    """
    YOLO-World detector with SAHI slicing support.

    Uses Ultralytics YOLO-World models for open-vocabulary detection
    and SAHI for sliced inference on large images.
    """

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._model = None
        self._sahi_enabled = True

    def load(self) -> None:
        """Load YOLO-World model."""
        if self._model is not None:
            return  # Already loaded

        try:
            import os
            # 强制离线模式，禁止从网络下载
            os.environ['YOLO_OFFLINE'] = '1'
            os.environ['ULTRALYTICS_OFFLINE'] = '1'

            from ultralytics import YOLO

            model_name = getattr(settings, 'YOLO_WORLD_MODEL', 'yolo11l-world.pt')
            logger.info(f"Loading YOLO-World model: {model_name}")

            # Check if model_name is a local file path
            from pathlib import Path
            model_path = Path(model_name)
            if model_path.exists() and model_path.is_file():
                logger.info(f"Using local model file: {model_path}")
                # Initialize YOLO model from local file
                self._model = YOLO(str(model_path))
            else:
                # Model will be auto-downloaded to ~/.cache/ultralytics/ on first use
                logger.info(f"Model will be downloaded if not cached: {model_name}")
                self._model = YOLO(model_name)

            # Move to specified device
            if self.device != 'cpu':
                self._model.to(self.device)

            # Get SAHI configuration
            self._slice_height = getattr(settings, 'SAHI_SLICE_HEIGHT', 640)
            self._slice_width = getattr(settings, 'SAHI_SLICE_WIDTH', 640)
            self._overlap_height_ratio = getattr(settings, 'SAHI_OVERLAP_HEIGHT_RATIO', 0.2)
            self._overlap_width_ratio = getattr(settings, 'SAHI_OVERLAP_WIDTH_RATIO', 0.2)

            logger.info(f"YOLO-World loaded successfully on {self.device}")
            logger.info(f"SAHI config: slice={self._slice_height}x{self._slice_width}, "
                       f"overlap={self._overlap_height_ratio}x{self._overlap_width_ratio}")

        except ImportError as e:
            raise ImportError(
                "Required packages not installed. Run:\n"
                "  pip install ultralytics>=8.3.0 sahi>=0.11.18"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise

    def _parse_classes_from_prompt(self, prompt: str) -> List[str]:
        """
        Parse class names from prompt string.

        Supports formats:
        - "person . car . dog"
        - "person, car, dog"
        - "person; car; dog"
        """
        import re
        parts = re.split(r'[.。,，;；]', prompt)
        classes = [p.strip() for p in parts if p.strip()]
        return classes

    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[RawDetection]:
        """
        Run YOLO-World detection with SAHI slicing.

        Args:
            image: BGR image (OpenCV format)
            prompt: Text prompt with class names (e.g., "person . car . dog")
            box_threshold: Confidence threshold for detections
            text_threshold: Not used for YOLO-World (kept for interface compatibility)

        Returns:
            List of RawDetection objects
        """
        if image is None or image.size == 0:
            logger.warning("predict() received empty image, skipping.")
            return []
        if image.ndim != 3 or image.shape[2] != 3:
            logger.warning(f"predict() unexpected image shape {image.shape}, skipping.")
            return []
        if image.shape[0] < 32 or image.shape[1] < 32:
            logger.warning(f"predict() image too small {image.shape}, skipping.")
            return []

        # Parse class names from prompt
        classes = self._parse_classes_from_prompt(prompt)
        if not classes:
            logger.warning("No valid classes found in prompt, skipping.")
            return []

        # Set classes for YOLO-World model
        self._model.set_classes(classes)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Decide whether to use SAHI based on image size
        use_sahi = (
            self._sahi_enabled and
            (h > self._slice_height * 1.5 or w > self._slice_width * 1.5)
        )

        if use_sahi:
            return self._predict_with_sahi(image_rgb, box_threshold, classes)
        else:
            return self._predict_direct(image_rgb, box_threshold, classes)

    def _predict_direct(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float,
        classes: List[str],
    ) -> List[RawDetection]:
        """Direct inference without SAHI slicing."""
        try:
            # Run inference
            results = self._model.predict(
                image_rgb,
                conf=conf_threshold,
                verbose=False,
                device=self.device,
            )

            detections = []
            if results and len(results) > 0:
                result = results[0]

                # Extract boxes, scores, and class indices
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, score, cls_id in zip(boxes, scores, class_ids):
                        x1, y1, x2, y2 = box
                        label = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"

                        detections.append(
                            RawDetection(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                                score=float(score),
                                label=label,
                            )
                        )

            return detections

        except Exception as e:
            logger.error(f"YOLO-World direct inference error: {e}")
            return []

    def _predict_with_sahi(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float,
        classes: List[str],
    ) -> List[RawDetection]:
        """Inference with SAHI slicing for better small object detection."""
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            # Wrap YOLO model for SAHI
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='ultralytics',
                model=self._model,
                confidence_threshold=conf_threshold,
                device=self.device,
            )

            # Run sliced prediction
            result = get_sliced_prediction(
                image_rgb,
                detection_model,
                slice_height=self._slice_height,
                slice_width=self._slice_width,
                overlap_height_ratio=self._overlap_height_ratio,
                overlap_width_ratio=self._overlap_width_ratio,
                verbose=0,
            )

            # Convert SAHI results to RawDetection
            detections = []
            for obj in result.object_prediction_list:
                bbox = obj.bbox
                x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                score = obj.score.value

                # Map category_id to class name
                cls_id = obj.category.id
                label = classes[cls_id] if cls_id < len(classes) else obj.category.name

                detections.append(
                    RawDetection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        score=float(score),
                        label=label,
                    )
                )

            logger.debug(f"SAHI detected {len(detections)} objects")
            return detections

        except ImportError:
            logger.warning("SAHI not available, falling back to direct inference")
            return self._predict_direct(image_rgb, conf_threshold, classes)
        except Exception as e:
            logger.error(f"SAHI inference error: {e}, falling back to direct inference")
            return self._predict_direct(image_rgb, conf_threshold, classes)
