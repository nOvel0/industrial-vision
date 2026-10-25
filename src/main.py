import logging
import time
from pathlib import Path

import yaml
import cv2

from src.capture.video_capture import VideoSource
from src.monitoring.fps import FPSMeter
from src.inference.detector import Detector
from src.inference.visualization import draw_detections, draw_zone
from src.zones.geometry import polygon_to_pixels
from src.tracking.tracker import Tracker
from src.events.event_engine import ZoneEventEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_NAME = "Industrial Vision"

logger = logging.getLogger(__name__)

def main() -> None:
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                logs_dir / "application.log",
                encoding="utf-8",
            ),
        ],
    )

    config_path = PROJECT_ROOT / "configs" / "configuration.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    camera_config = config["camera"]
    source = camera_config["source"]
    reconnect_sec = float(camera_config.get("reconnect_sec", 2))

    if reconnect_sec <= 0:
        raise ValueError("camera.reconnect_sec должен быть больше нуля")

    zone_id = "restricted_01"
    zone_polygon = config["zones"][zone_id]

    if len(zone_polygon) < 3:
        raise ValueError("Для зоны необходимо минимум три точки")

    # Относительный путь к видео считаем от корня проекта.
    if isinstance(source, str) and "://" not in source:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        source = str(source_path)

    model_config = config["models"]

    model_path = Path(model_config["detector_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    try:
        detector = Detector(
            model_path=model_path,
            confidence=float(model_config["confidence"]),
            imgsz=int(model_config["imgsz"]),
            device=str(model_config["device"]),
            classes=model_config["classes"],
        )
    except Exception:
        logger.exception("Не удалось подготовить детектор")
        raise

    tracker = Tracker(detector)

    event_config = config["events"]

    zone_events = ZoneEventEngine(
        zone_id=zone_id,
        confirmation_n=int(event_config["confirmation_n"]),
        confirmation_m=int(event_config["confirmation_m"]),
        cooldown_sec=float(event_config["cooldown_sec"]),
    )

    video = VideoSource(source)
    fps_meter = FPSMeter()

    connected = False
    retry_at = 0.0

    logger.info("Запуск камеры %s", camera_config["id"])

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        last_fps_log = time.perf_counter()
        while True:
            if not connected:
                if time.perf_counter() >= retry_at:
                    try:
                        video.open()
                    except RuntimeError as error:
                        logger.error("%s", error)
                        retry_at = time.perf_counter() + reconnect_sec
                        logger.info(
                            "Следующая попытка подключения через %.1f секунд",
                            reconnect_sec,
                        )
                    else:
                        connected = True
                        tracker.reset()
                        zone_events.reset()
                        fps_meter.reset()
                        last_fps_log = time.perf_counter()

                if not connected:
                    if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                        break
                    continue

            try:
                frame = video.read()
            except RuntimeError as error:
                logger.error("Видеопоток потерян: %s", error)
                video.close()
                connected = False
                retry_at = time.perf_counter() + reconnect_sec
                continue

            if frame is None:
                logger.info("Чтение видеофайла завершено: конец файла или ошибка декодирования")
                break

            inference_started = time.perf_counter()

            try:
                detections = tracker.update(frame)
            except Exception:
                logger.exception("Ошибка детекции и tracking")
                raise

            inference_ms = (
                time.perf_counter() - inference_started
            ) * 1000

            height, width = frame.shape[:2]

            polygon_px = polygon_to_pixels(
                zone_polygon,
                width,
                height,
            )

            events = zone_events.update(detections, polygon_px)

            for event in events:
                logger.warning(
                    "Событие: %s | камера: %s | зона: %s | "
                    "track_id: %d | confidence: %.2f",
                    event["event_type"],
                    camera_config["id"],
                    event["zone_id"],
                    event["track_id"],
                    event["confidence"],
                )

            draw_detections(frame, detections)
            draw_zone(frame, detections, polygon_px)

            fps = fps_meter.tick()
            now = time.perf_counter()

            if now - last_fps_log >= 1.0:
                logger.info(
                    "FPS обработки: %.1f | "
                    "детекция и tracking: %.1f мс | "
                    "объектов: %d",
                    fps,
                    inference_ms,
                    len(detections),
                )

                last_fps_log = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Detect + Track: {inference_ms:.1f} ms | Objects: {len(detections)}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        logger.info("Остановлено с клавиатуры")
    finally:
        video.close()
        cv2.destroyAllWindows()
        logger.info("Ресурсы освобождены")


if __name__ == "__main__":
    main()
