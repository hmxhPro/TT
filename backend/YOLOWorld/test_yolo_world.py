"""
Test script for YOLO-World + SAHI detector.

Usage:
    python test_yolo_world.py [--image path/to/image.jpg] [--prompt "person . car"]
"""

import sys
import argparse
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import cv2
import numpy as np
from app.core.config import settings
from app.core.logging import logger


def test_yolo_world_detector(image_path: str, prompt: str):
    """Test YOLO-World detector on a single image."""

    # Temporarily set detection model to yolo_world
    original_model = settings.DETECTION_MODEL
    settings.DETECTION_MODEL = "yolo_world"

    try:
        # Import detector
        from app.services.detector import get_detector

        logger.info("Loading YOLO-World detector...")
        detector = get_detector()

        # Load test image
        logger.info(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        h, w = image.shape[:2]
        logger.info(f"Image size: {w}x{h}")

        # Run detection
        logger.info(f"Running detection with prompt: '{prompt}'")
        detections = detector.predict(
            image=image,
            prompt=prompt,
            box_threshold=0.25,
            text_threshold=0.25,
        )

        # Print results
        logger.info(f"Found {len(detections)} detections:")
        for i, det in enumerate(detections, 1):
            logger.info(
                f"  {i}. {det.label}: {det.score:.3f} "
                f"at [{det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f}]"
            )

        # Visualize results
        output_image = image.copy()
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label_text = f"{det.label}: {det.score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_image, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(
                output_image, label_text, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # Save output
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_yolo_world_output.jpg"
        cv2.imwrite(str(output_path), output_image)
        logger.info(f"Saved output to: {output_path}")

        return detections

    finally:
        # Restore original model setting
        settings.DETECTION_MODEL = original_model


def main():
    parser = argparse.ArgumentParser(description="Test YOLO-World detector")
    parser.add_argument(
        "--image",
        type=str,
        default="./uploads/test.jpg",
        help="Path to test image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person . car . dog",
        help="Detection prompt (classes separated by . or ,)"
    )

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        print("Please provide a valid image path using --image")
        return 1

    try:
        detections = test_yolo_world_detector(args.image, args.prompt)
        print(f"\n✓ Test completed successfully! Found {len(detections)} objects.")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
