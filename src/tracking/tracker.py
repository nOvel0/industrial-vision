"""Отслеживание объектов с помощью ByteTrack."""

import numpy as np

from src.inference.detector import Detector
from src.inference.types import Detection


class Tracker:
    def __init__(self, detector: Detector):
        self.detector = detector

    def update(self, frame: np.ndarray) -> list[Detection]:
        result = self.detector.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.detector.confidence,
            imgsz=self.detector.imgsz,
            device=self.detector.device,
            classes=self.detector.classes,
            verbose=False,
        )[0]

        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            track_id = (
                int(box.id.item())
                if box.id is not None
                else None
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(result.names[class_id]),
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    track_id=track_id,
                )
            )

        return detections

    def reset(self) -> None:
        """Сбросить tracking после переподключения источника."""
        predictor = self.detector.model.predictor

        if predictor is not None:
            for tracker in getattr(predictor, "trackers", []):
                tracker.reset()