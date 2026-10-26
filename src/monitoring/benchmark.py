import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import psutil


class Benchmark:
    def __init__(self):
        self.rows = []
        self.resources = []
        self.count = 0
        self.model_ms = dict(preprocess=0.0, inference=0.0, postprocess=0.0)
        self.process = psutil.Process()
        psutil.cpu_percent()

    def start(self):
        self.started = time.perf_counter()
        self.model_ms = dict(preprocess=0.0, inference=0.0, postprocess=0.0)

    def record_model(self, speed):
        for name in self.model_ms:
            self.model_ms[name] += speed[name]

    def finish(self, capture_ms, post_ms, event_ms):
        total = (time.perf_counter() - self.started) * 1000
        row = {
            "capture": capture_ms,
            "preprocessing": self.model_ms["preprocess"],
            "inference": self.model_ms["inference"],
            "postprocessing": self.model_ms["postprocess"] + post_ms,
            "event_processing": event_ms,
        }
        row["other"] = max(0.0, total - sum(row.values()))
        row["total_pipeline"] = total
        self.count += 1
        if self.count > 5:
            self.rows.append(row)

    def sample_resources(self):
        temperatures = psutil.sensors_temperatures()
        values = [item.current for group in temperatures.values() for item in group]
        sample = {
            "cpu_percent": psutil.cpu_percent(),
            "process_ram_mb": self.process.memory_info().rss / 1024**2,
            "system_ram_percent": psutil.virtual_memory().percent,
            "max_sensor_temperature_c": max(values) if values else None,
            "disk_free_gb": shutil.disk_usage(Path.cwd()).free / 1024**3,
            "disk_used_gb": shutil.disk_usage(Path.cwd()).used / 1024**3,
        }
        self.resources.append(sample)
        logging.getLogger("system_monitoring").info("Ресурсы: %s", sample)

    def save(self, path):
        if not self.rows:
            return
        metrics = {}
        for name in self.rows[0]:
            values = [row[name] for row in self.rows]
            metrics[name] = {
                "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": float(np.max(values)),
            }
        report = {
            "frames": len(self.rows),
            "warmup_frames": 5,
            "fps": 1000 / metrics["total_pipeline"]["mean_ms"],
            "metrics": metrics,
            "resources": self.resources,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


benchmark = Benchmark()
