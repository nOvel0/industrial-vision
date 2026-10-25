"""Временное подтверждение нахождения человека в зоне."""

import time
from collections import deque

from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon


class ZoneEventEngine:
    def __init__(self, zone_id: str, confirmation_n: int = 3, confirmation_m: int = 5, cooldown_sec: float = 10):
        if not 1 <= confirmation_n <= confirmation_m:
            raise ValueError("Необходимо 1 <= confirmation_n <= confirmation_m")

        if cooldown_sec < 0:
            raise ValueError("cooldown_sec не может быть отрицательным")

        self.zone_id = zone_id
        self.n = confirmation_n
        self.m = confirmation_m
        self.cooldown_sec = cooldown_sec

        self.history: dict[int, deque] = {}
        self.last_event: dict[int, float] = {}
        self.last_seen: dict[int, float] = {}

    def update(self, detections: list[Detection], polygon_px: list[tuple[int, int]]) -> list[dict]:
        now = time.perf_counter()

        # Люди с известным ID, обнаруженные на текущем кадре.
        people = {}

        for obj in detections:
            if obj.class_name == "person" and obj.track_id is not None:
                people[obj.track_id] = obj

        for track_id in people:
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.m)

            self.last_seen[track_id] = now

        events = []

        # Обновляем также отсутствующих людей:
        # пропуск детекции считается False.
        for track_id, window in list(self.history.items()):
            obj = people.get(track_id)

            inside = (
                obj is not None
                and point_in_polygon(bottom_center(obj), polygon_px)
            )

            window.append(inside)

            confirmed = (
                len(window) == self.m
                and sum(window) >= self.n
            )

            previous_event = self.last_event.get(track_id)
            cooldown_finished = (
                previous_event is None
                or now - previous_event >= self.cooldown_sec
            )

            if inside and confirmed and cooldown_finished:
                events.append({
                    "event_type": "person_in_restricted_zone",
                    "zone_id": self.zone_id,
                    "track_id": track_id,
                    "confidence": obj.confidence,
                })

                self.last_event[track_id] = now

            # Удаляем давно исчезнувшие траектории из памяти.
            retention_sec = max(60.0, self.cooldown_sec)

            if now - self.last_seen[track_id] > retention_sec:
                del self.history[track_id]
                self.last_event.pop(track_id, None)
                del self.last_seen[track_id]

        return events

    def reset(self) -> None:
        self.history.clear()
        self.last_event.clear()
        self.last_seen.clear()