import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import ultralytics
from ultralytics import YOLO

weights = Path(sys.argv[1])
video_path = sys.argv[2]
model = YOLO(str(weights), task="detect")
video = cv2.VideoCapture(video_path)
process = psutil.Process()
times = []
ram = []

try:
    for index in range(205):
        ok, frame = video.read()
        if not ok:
            break
        started = time.perf_counter()
        model.predict(
            frame, imgsz=640, conf=0.40, classes=[0, 1],
            device="cpu", rect=False, verbose=False,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if index >= 5:
            times.append(elapsed)
            ram.append(process.memory_info().rss / 1024**2)
finally:
    video.release()

if not times:
    raise RuntimeError("Для замера нужно видео длиннее пяти кадров")

report = {
    "scope": "single PPE model, predict(), without capture or tracking",
    "weights": str(weights),
    "video": video_path,
    "imgsz": 640,
    "rect": False,
    "confidence": 0.40,
    "classes": [0, 1],
    "frames": len(times),
    "warmup_frames": 5,
    "model_size_mb": weights.stat().st_size / 1024**2,
    "mean_ms": float(np.mean(times)),
    "p95_ms": float(np.percentile(times, 95)),
    "max_ms": float(np.max(times)),
    "fps": 1000 / float(np.mean(times)),
    "peak_process_ram_mb": max(ram),
    "torch": torch.__version__,
    "ultralytics": ultralytics.__version__,
    "device": "cpu",
}
output = Path("runs/benchmark") / f"{weights.stem}_{weights.suffix[1:]}.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("Saved:", output)
