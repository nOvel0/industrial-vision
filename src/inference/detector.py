"""Детектор объектов на основе предобученной YOLO."""

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.inference.types import Detection


logger = logging.getLogger(__name__)

class Detector:
    def __init__(
            self, 
            model_path: str | Path,
            confidence: float = 0.5, 
            imgsz: int = 640, 
            device: str = "cpu", 
            classes: list[int] | None = None
    ):
        if not 0 < confidence <= 1:
            raise ValueError("Порог confidence должен быть больше 0 и не больше 1")

        if imgsz <= 0:
            raise ValueError("Размер входа модели должен быть больше нуля")

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(f"Файл весов модели не найден: {model_path}")

        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.classes = classes

        try:
            self.model = YOLO(str(model_path), task="detect")
        except Exception as error:
            raise RuntimeError(f"Не удалось загрузить модель: {model_path}") from error

        logger.info(
            "Модель загружена: %s | устройство: %s | confidence: %.2f",
            model_path,
            device,
            confidence,
        )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False
        )[0]

        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(result.names[class_id]),
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections