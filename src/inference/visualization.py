"""Отображение результатов детекции на кадре."""

import cv2
import numpy as np

from src.inference.types import Detection


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for detection in detections:
        color = (
            (0, 255, 0) # зеленый
            if detection.class_name == "person"
            else (0, 0, 255) # красный 
        )

        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (detection.x1, max(20, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )