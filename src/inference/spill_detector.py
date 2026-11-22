"""Сегментация разлива моделью YOLO."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.monitoring.benchmark import benchmark
from src.zones.geometry import polygon_to_pixels


class SpillDetector:
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        imgsz: int,
        device: str,
        polygon: list[list[float]] | None,
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"Файл весов разлива не найден: {model_path}")
        if not 0 < confidence <= 1 or imgsz <= 0:
            raise ValueError("Для модели разлива нужны confidence от 0 до 1 и положительный imgsz")
        if polygon is not None and (
            len(polygon) < 3
            or any(len(point) != 2 or not all(0 <= value <= 1 for value in point) for point in polygon)
        ):
            raise ValueError("spill.polygon должен содержать минимум три точки [x, y] в диапазоне [0, 1]")

        self.model = YOLO(str(model_path))
        if self.model.task != "segment":
            raise ValueError("Для разлива нужна модель сегментации")
        self.classes = [class_id for class_id, name in self.model.names.items() if name == "spill"]
        if not self.classes:
            raise ValueError("В модели разлива нет класса spill")

        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.polygon = polygon

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, float | None]:
        height, width = frame.shape[:2]
        x, y = 0, 0
        crop = frame
        zone_mask = None

        if self.polygon is not None:
            contour = np.array(polygon_to_pixels(self.polygon, width, height), dtype=np.int32)
            x, y, crop_width, crop_height = cv2.boundingRect(contour)
            crop = frame[y:y + crop_height, x:x + crop_width]

            # Закрываем пиксели за границей зоны.
            zone_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            contour -= (x, y)
            cv2.fillPoly(zone_mask, [contour], 1)
            crop = cv2.bitwise_and(crop, crop, mask=zone_mask)

        result = self.model.predict(
            crop,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            retina_masks=True,
            verbose=False,
        )[0]
        benchmark.record_model(result.speed)

        mask = np.zeros((height, width), dtype=np.uint8)
        if result.masks is None or result.boxes is None:
            return mask, None

        masks = result.masks.data.cpu().numpy() > 0
        if zone_mask is not None:
            masks &= zone_mask.astype(bool)
        visible = masks.any(axis=(1, 2))
        if not visible.any():
            return mask, None

        # Переносим маски в координаты полного кадра.
        crop_height, crop_width = crop.shape[:2]
        mask[y:y + crop_height, x:x + crop_width] = masks.any(axis=0) * 255
        confidence = float(result.boxes.conf.cpu().numpy()[visible].max())
        return mask, confidence
