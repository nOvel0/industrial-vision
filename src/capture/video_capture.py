"""Получение кадров из видеофайла или камеры."""

import logging
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, source: str | int):
        self.source = source
        self.cap: cv2.VideoCapture | None = None

        # Локальный файл отличаем от индекса камеры и сетевого адреса
        self.is_file = (
            isinstance(source, str)
            and "://" not in source
        )

    def open(self) -> None:
        self.close()

        if self.is_file and not Path(self.source).is_file():
            raise RuntimeError(f"Видеофайл не найден: {self.source}")

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.close()
            raise RuntimeError(f"Не удалось открыть источник видео: {self.source}")

        logger.info("Источник видео открыт: %s", self.source)

    def read(self) -> np.ndarray | None:
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Источник видео не открыт")

        ok, frame = self.cap.read()

        if ok and frame is not None:
            return frame

        if self.is_file:
            return None

        raise RuntimeError(
            f"Не удалось получить кадр с камеры: {self.source}"
        )

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None