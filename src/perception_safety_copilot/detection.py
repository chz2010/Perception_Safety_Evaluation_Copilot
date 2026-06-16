from __future__ import annotations

from typing import Any

import numpy as np

from .evaluation import Detection


def load_yolo_model(model_name: str = "yolov8n.pt") -> Any:
    from ultralytics import YOLO

    return YOLO(model_name)


def run_yolo_detection(
    model: Any,
    image_rgb: np.ndarray,
    confidence_threshold: float,
    *,
    image_id: str | None = None,
    model_name: str | None = None,
) -> list[Detection]:
    results = model.predict(image_rgb, conf=confidence_threshold, verbose=False)
    if not results:
        return []

    result = results[0]
    detections: list[Detection] = []
    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = str(names.get(cls_id, cls_id))
        confidence = float(box.conf.item())
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                bbox_xyxy=xyxy,
                image_id=image_id,
                model_name=model_name,
            )
        )

    return detections


def detections_to_records(detections: list[Detection]) -> list[dict[str, Any]]:
    return [
        {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox_xyxy": list(detection.bbox_xyxy),
            "image_id": detection.image_id,
            "model_name": detection.model_name,
        }
        for detection in detections
    ]


def draw_detections(image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    import cv2

    annotated = image_rgb.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection.bbox_xyxy]
        label = f"{detection.label} {detection.confidence:.2f}"
        color = (30, 180, 80) if detection.confidence >= 0.5 else (255, 165, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_w, text_h = text_size
        y_text = max(y1 - 8, text_h + 8)
        cv2.rectangle(annotated, (x1, y_text - text_h - 8), (x1 + text_w + 8, y_text + 4), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, y_text - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
