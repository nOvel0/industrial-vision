"""Отображение результатов детекции на кадре."""

import cv2
import numpy as np

from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon


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

        id_text = (
            f"ID: {detection.track_id} "
            if detection.track_id is not None
            else ""
        )

        label = (
            f"{id_text}"
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

def draw_zone(frame: np.ndarray, detections: list[Detection], polygon: list[tuple[int, int]]) -> None:
    contour = np.array(polygon, dtype=np.int32)

    cv2.polylines(
        frame,
        [contour],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
    )

    for detection in detections:
        if detection.class_name != "person":
            continue

        point = bottom_center(detection)
        inside = point_in_polygon(point, polygon)

        color = (0, 0, 255) if inside else (0, 255, 0)

        cv2.circle(
            frame,
            point,
            radius=5,
            color=color,
            thickness=-1,
        )