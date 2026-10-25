"""Сохранение событий и подтверждающих кадров."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.events.types import Event


logger = logging.getLogger("ai_events")


class EventStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        event: Event,
        frame: np.ndarray | None = None,
    ) -> Path:
        timestamp = datetime.fromisoformat(event.timestamp)

        event_dir = (
            self.root
            / timestamp.strftime("%Y")
            / timestamp.strftime("%m")
            / timestamp.strftime("%d")
            / event.event_id
        )

        event_dir.mkdir(parents=True, exist_ok=False)

        if frame is not None:
            snapshot_path = event_dir / "snapshot.jpg"

            if not cv2.imwrite(str(snapshot_path), frame):
                raise OSError(
                    f"Не удалось сохранить кадр: {snapshot_path}"
                )

            event.snapshot = snapshot_path.as_posix()

        json_path = event_dir / "event.json"

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(
                asdict(event),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        logger.info(
            "Событие сохранено: %s | тип: %s | камера: %s | "
            "track_id: %s | JSON: %s",
            event.event_id,
            event.event_type,
            event.camera_id,
            event.track_id,
            json_path,
        )

        return json_path