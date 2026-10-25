"""Подтверждение событий отсутствия каски и жилета."""

import time
from collections import deque


class PPEEventEngine:
    def __init__(self, n: int, m: int, cooldown_sec: float):
        if not 1 <= n <= m:
            raise ValueError("Необходимо 1 <= n <= m")

        if cooldown_sec < 0:
            raise ValueError("cooldown_sec не может быть отрицательным")

        self.n = n
        self.m = m
        self.cooldown_sec = cooldown_sec

        self.history = {}
        self.last_event = {}

    def update(self, ppe_results: list[dict]) -> list[dict]:
        now = time.perf_counter()
        states = {}

        for result in ppe_results:
            track_id = result["person"].track_id

            for field, event_type in (
                ("has_helmet", "ppe_helmet_missing"),
                ("has_vest", "ppe_vest_missing"),
            ):
                status = result[field]

                # None означает, что проверка недоступна.
                if status is None:
                    continue

                key = (track_id, event_type)
                states[key] = not status

        # Если человек исчез или проверка недоступна,
        # старое подтверждение сбрасывается.
        for key in list(self.history):
            if key not in states:
                del self.history[key]

        events = []

        for key, missing in states.items():
            if key not in self.history:
                self.history[key] = deque(maxlen=self.m)

            window = self.history[key]
            window.append(missing)

            confirmed = (
                len(window) == self.m
                and sum(window) >= self.n
            )

            previous = self.last_event.get(key)
            cooldown_finished = (
                previous is None
                or now - previous >= self.cooldown_sec
            )

            if missing and confirmed and cooldown_finished:
                track_id, event_type = key

                events.append({
                    "event_type": event_type,
                    "track_id": track_id,
                })

                self.last_event[key] = now

        # Удаляем старые записи cooldown из памяти.
        retention_sec = max(60.0, self.cooldown_sec)

        for key, timestamp in list(self.last_event.items()):
            if now - timestamp > retention_sec:
                del self.last_event[key]

        return events

    def reset(self) -> None:
        self.history.clear()
        self.last_event.clear()