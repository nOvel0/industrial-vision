"""Подтверждение событий людей и транспорта."""

import time
from collections import deque

from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon


VEHICLE_CLASSES = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


class ZoneEventEngine:
    def __init__(
        self,
        zone_id: str,
        confirmation_n: int = 3,
        confirmation_m: int = 5,
        cooldown_sec: float = 10,
    ):
        if not 1 <= confirmation_n <= confirmation_m:
            raise ValueError("Необходимо 1 <= confirmation_n <= confirmation_m")

        if cooldown_sec < 0:
            raise ValueError("cooldown_sec не может быть отрицательным")

        self.zone_id = zone_id
        self.n = confirmation_n
        self.m = confirmation_m
        self.cooldown_sec = cooldown_sec

        self.history = {}
        self.last_event = {}

    def update(
        self,
        detections: list[Detection],
        polygon_px: list[tuple[int, int]],
    ) -> list[dict]:
        now = time.perf_counter()
        states = {}

        for obj in detections:
            if obj.track_id is None:
                continue

            inside = point_in_polygon(
                bottom_center(obj),
                polygon_px,
            )

            if obj.class_name == "person":
                states[(obj.track_id, "person_detected")] = (obj, True)

                states[
                    (obj.track_id, "person_in_restricted_zone")
                ] = (obj, inside)

            elif obj.class_name in VEHICLE_CLASSES:
                states[
                    (obj.track_id, "vehicle_in_restricted_zone")
                ] = (obj, inside)

        for key in states:
            if key not in self.history:
                self.history[key] = deque(maxlen=self.m)

        events = []

        for key, window in list(self.history.items()):
            obj, active = states.get(key, (None, False))
            window.append(active)

            confirmed = (
                len(window) == self.m
                and sum(window) >= self.n
            )

            previous = self.last_event.get(key)
            cooldown_finished = (
                previous is None
                or now - previous >= self.cooldown_sec
            )

            if active and confirmed and cooldown_finished:
                track_id, event_type = key

                events.append({
                    "event_type": event_type,
                    "track_id": track_id,
                    "confidence": obj.confidence,
                    "zone_id": (
                        None
                        if event_type == "person_detected"
                        else self.zone_id
                    ),
                    "object": obj,
                })

                self.last_event[key] = now

            # Пустую историю больше хранить не нужно.
            if not any(window):
                del self.history[key]

        # Записи cooldown удаляем после истечения срока.
        retention_sec = max(60.0, self.cooldown_sec)

        for key, timestamp in list(self.last_event.items()):
            if now - timestamp > retention_sec:
                del self.last_event[key]

        return events

    def reset(self) -> None:
        self.history.clear()
        self.last_event.clear()