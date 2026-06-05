"""
app/api/image_detect.py
-----------------------
Still-image detection (REQ1): zero-shot YOLOE OR a chosen trained model.

POST /api/image-detect                 -> detect on one/many uploaded images
GET  /api/image-detect/{batch}/{file}  -> serve an annotated result image

Exactly one of `model_id` (use a trained model's baked-in classes) or
`class_names` (zero-shot open-vocabulary) must be supplied.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging import logger
from app.db.models import TrainedModelRecord
from app.db.session import AsyncSessionLocal
from app.models.schemas import ImageDetectResponse, ImageDetectResultItem

router = APIRouter()


def _reject_unsafe(component: str) -> None:
    if "/" in component or "\\" in component or ".." in component:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "非法的标识符。")


def _imgdet_dir(batch_id: str):
    return settings.RESULTS_DIR / settings.IMGDET_SUBDIR / batch_id


@router.post(
    "/image-detect",
    response_model=ImageDetectResponse,
    summary="图片检测（零样本 YOLOE 或选定的已训练模型）",
)
async def image_detect(
    files: List[UploadFile] = File(..., description="一张或多张图片"),
    model_id: Optional[str] = Form(None),
    class_names: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
) -> ImageDetectResponse:
    import asyncio
    from app.services.image_detector import annotate_image, detect_with_model, detect_zeroshot

    model_id = (model_id or "").strip() or None
    class_names = (class_names or "").strip() or None
    if bool(model_id) == bool(class_names):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "请二选一：提供 model_id（使用已训练模型）或 class_names（零样本）。",
        )
    if conf is not None:
        conf = max(0.0, min(1.0, float(conf)))

    # Resolve detection mode.
    weights_path: Optional[str] = None
    classes: List[str] = []
    mode: str
    resp_class_names: List[str] = []
    if model_id:
        mode = "model"
        try:
            async with AsyncSessionLocal() as session:
                rec = await session.get(TrainedModelRecord, model_id)
        except Exception as exc:
            logger.warning(f"image_detect model lookup failed: {exc}")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "所选模型不存在。")
        weights_path = rec.weights_path
        if isinstance(rec.class_names, dict):
            resp_class_names = [str(v) for v in rec.class_names.values()]
    else:
        mode = "zeroshot"
        classes = [c.strip() for c in class_names.replace("，", ",").split(",") if c.strip()]
        if not classes:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "零样本检测至少需要一个类别名。")
        resp_class_names = classes

    batch_id = uuid.uuid4().hex
    out_dir = _imgdet_dir(batch_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[ImageDetectResultItem] = []
    for idx, f in enumerate(files):
        raw = await f.read()
        if not raw:
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.info(f"image_detect: skipping undecodable file {f.filename}")
            continue
        h, w = image.shape[:2]

        try:
            if mode == "model":
                detections = await asyncio.to_thread(detect_with_model, weights_path, image, conf)
            else:
                detections = await asyncio.to_thread(detect_zeroshot, classes, image, conf)
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        annotated = await asyncio.to_thread(annotate_image, image, detections)
        out_name = f"{idx}.jpg"
        cv2.imwrite(str(out_dir / out_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        results.append(
            ImageDetectResultItem(
                image_index=idx,
                filename=f.filename or out_name,
                width=w,
                height=h,
                detections=detections,
                annotated_url=f"/api/image-detect/{batch_id}/{out_name}",
            )
        )

    if not results:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "没有可解码的图片。")

    logger.info(
        f"image_detect batch={batch_id} mode={mode} images={len(results)} "
        f"classes={resp_class_names}"
    )
    return ImageDetectResponse(
        batch_id=batch_id,
        mode=mode,
        model_id=model_id,
        class_names=resp_class_names,
        results=results,
    )


@router.get(
    "/image-detect/{batch_id}/{filename}",
    response_class=FileResponse,
    summary="获取图片检测的标注结果图",
)
async def get_imgdet_result(batch_id: str, filename: str):
    _reject_unsafe(batch_id)
    _reject_unsafe(filename)
    p = _imgdet_dir(batch_id) / filename
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "结果图不存在。")
    return FileResponse(path=str(p), media_type="image/jpeg")
