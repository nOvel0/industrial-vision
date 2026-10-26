"""Прогон первых 205 кадров без открытия окна; 5 кадров прогрева."""

import json

import yaml

import src.main as vision


frames = 0


def next_frame(delay):
    global frames
    frames += 1
    return ord("q") if frames >= 205 else -1


# Рисование рамок сохраняется; вывод окна отключён для headless-замера.
vision.cv2.namedWindow = lambda *args, **kwargs: None
vision.cv2.imshow = lambda *args, **kwargs: None
vision.cv2.waitKey = next_frame

vision.main()

if not vision.benchmark.rows:
    raise RuntimeError("Нет измерений: требуется больше пяти кадров")

path = vision.PROJECT_ROOT / "runs/benchmark/pipeline.json"
report = json.loads(path.read_text(encoding="utf-8"))
config = yaml.safe_load(
    (vision.PROJECT_ROOT / "configs/configuration.yaml").read_text(encoding="utf-8")
)
report["mode"] = "headless; drawing retained, namedWindow/imshow/waitKey disabled"
report["api_enabled"] = True
report["spill_enabled"] = config["spill"].get("enabled", False)
report["source"] = config["camera"]["source"]
path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("Measured frames:", report["frames"])
print("Pipeline FPS:", round(report["fps"], 2))
print("Saved:", path)
