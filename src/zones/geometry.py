"""Геометрия контролируемых зон."""

import cv2
import numpy as np

from src.inference.types import Detection


def polygon_to_pixels(polygon: list[list[float]], width: int, height: int) -> list[tuple[int, int]]:
    """Перевести нормированные координаты зоны в пиксели."""
    return [
        (
            round(x * (width - 1)),
            round(y * (height - 1)),
        )
        for x, y in polygon
    ]


def bottom_center(detection: Detection) -> tuple[int, int]:
    """Нижняя центральная точка рамки объекта."""
    return (
        (detection.x1 + detection.x2) // 2,
        detection.y2,
    )


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    """Проверить попадание точки внутрь зоны или на её границу."""
    contour = np.array(polygon, dtype=np.int32)

    return cv2.pointPolygonTest(contour, point, False) >= 0