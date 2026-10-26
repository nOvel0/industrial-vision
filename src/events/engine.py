"""Общее создание и сохранение подтверждённых событий."""

import numpy as np

from src.events.types import Event
from src.inference.types import Detection
from src.storage.event_store import EventStore


class EventEngine:
    def __init__(self, camera_id: str, store: EventStore):
        self.camera_id = camera_id
        self.store = store

    def emit(
        self,
        data: dict,
        frame: np.ndarray | None = None,
        obj: Detection | None = None,
    ) -> Event:
        coordinates = data.get("coordinates", {})

        if obj is not None and frame is not None:
            height, width = frame.shape[:2]

            coordinates = {
                "x": ((obj.x1 + obj.x2) / 2) / width,
                "y": obj.y2 / height,
                "bbox": [
                    obj.x1 / width,
                    obj.y1 / height,
                    obj.x2 / width,
                    obj.y2 / height,
                ],
            }

        event = Event(
            event_type=data["event_type"],
            camera_id=self.camera_id,
            confidence=data.get("confidence"),
            track_id=data.get("track_id"),
            zone_id=data.get("zone_id"),
            coordinates=coordinates,
        )

        self.store.save(event, frame)

        return event