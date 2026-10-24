"""Измерение фактической частоты обработки кадров."""

import time
from collections import deque

class FPSMeter:
    def __init__(self, window: int = 30):
        if window < 2:
            raise ValueError("Диапазон FPS должен быть не менее 2")

        self.times = deque(maxlen=window)

    def tick(self) -> float:
        self.times.append(time.perf_counter())

        if len(self.times) < 2:
            return 0.0

        dt = self.times[-1] - self.times[0]

        return (
            (len(self.times) - 1) / dt
            if dt > 0
            else 0.0
        )

    def reset(self) -> None:
        self.times.clear()