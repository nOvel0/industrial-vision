"""Подтверждение событий дыма и огня по нескольким кадрам."""

import time
from collections import deque

from src.inference.types import Detection


class FireSmokeEventEngine:
    def __init__(self, n: int, m: int, cooldown_sec: float):
        if not 1 <= n <= m:
            raise ValueError("Необходимо 1 <= n <= m")

        if cooldown_sec < 0:
            raise ValueError("cooldown_sec не может быть отрицательным")

        self.n = n
        self.m = m
        self.cooldown_sec = cooldown_sec

        self.history = {
            "smoke": deque(maxlen=m),
            "fire": deque(maxlen=m),
        }
        self.last_event = {}

    def update(self, detections: list[Detection]) -> list[dict]:
        now = time.perf_counter()
        events = []

        for class_name, window in self.history.items():
            objects = [
                obj
                for obj in detections
                if obj.class_name == class_name
            ]

            detected = bool(objects)
            window.append(detected)

            confirmed = (
                len(window) == self.m
                and sum(window) >= self.n
            )

            previous = self.last_event.get(class_name)
            cooldown_finished = (
                previous is None
                or now - previous >= self.cooldown_sec
            )

            if detected and confirmed and cooldown_finished:
                best = max(objects, key=lambda obj: obj.confidence)

                events.append({
                    "event_type": f"{class_name}_detected",
                    "confidence": best.confidence,
                })

                self.last_event[class_name] = now

        return events

    def reset(self) -> None:
        for window in self.history.values():
            window.clear()

        self.last_event.clear()