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
from src.ppe.analyzer import check_ppe
from src.events.ppe_events import PPEEventEngine
from src.events.fire_smoke_events import FireSmokeEventEngine

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

    ppe_config = config["ppe"]

    ppe_model_path = Path(ppe_config["model_path"])

    if not ppe_model_path.is_absolute():
        ppe_model_path = PROJECT_ROOT / ppe_model_path

    ppe_detector = Detector(
        model_path=ppe_model_path,
        confidence=float(ppe_config["confidence"]),
        imgsz=int(ppe_config["imgsz"]),
        device=str(model_config["device"]),
        classes=[0, 1],
    )

    min_person_height = int(ppe_config["min_person_height"])

    if min_person_height <= 0:
        raise ValueError("ppe.min_person_height должен быть больше нуля")

    ppe_events = PPEEventEngine(
        n=int(ppe_config["confirmation_n"]),
        m=int(ppe_config["confirmation_m"]),
        cooldown_sec=float(event_config["cooldown_sec"]),
    )

    fire_config = config["fire_smoke"]

    fire_model_path = Path(fire_config["model_path"])

    if not fire_model_path.is_absolute():
        fire_model_path = PROJECT_ROOT / fire_model_path

    fire_detector = Detector(
        model_path=fire_model_path,
        confidence=float(fire_config["confidence"]),
        imgsz=int(fire_config["imgsz"]),
        device=str(model_config["device"]),
        classes=[0, 1],
    )

    logger.info(
        "Классы модели дыма и огня: %s",
        fire_detector.model.names,
    )

    fire_events = FireSmokeEventEngine(
        n=int(fire_config["confirmation_n"]),
        m=int(fire_config["confirmation_m"]),
        cooldown_sec=float(fire_config["cooldown_sec"]),
    )

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
                        ppe_events.reset()
                        fire_events.reset()
                        frame_number = 0
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

            frame_number += 1

            inference_started = time.perf_counter()

            try:
                detections = tracker.update(frame)
                fire_detections = fire_detector.predict(frame)
                equipment = []

                frame_height, frame_width = frame.shape[:2]

                for person in detections:
                    if person.class_name != "person" or person.track_id is None:
                        continue

                    person_height = person.y2 - person.y1

                    if person_height < min_person_height:
                        continue

                    # Немного расширяем область вокруг человека.
                    padding = int(person_height * 0.10)

                    x1 = max(0, person.x1 - padding)
                    y1 = max(0, person.y1 - padding)
                    x2 = min(frame_width, person.x2 + padding)
                    y2 = min(frame_height, person.y2 + padding)

                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    items = ppe_detector.predict(crop)

                    # Возвращаем координаты рамок в систему полного кадра.
                    for item in items:
                        item.x1 += x1
                        item.x2 += x1
                        item.y1 += y1
                        item.y2 += y1

                    equipment.extend(items)
            except Exception:
                logger.exception("Ошибка обработки кадра моделями")
                raise

            inference_ms = (
                time.perf_counter() - inference_started
            ) * 1000

            height, width = frame.shape[:2]

            ppe_results = check_ppe(
                people=detections,
                equipment=equipment,
                min_person_height=min_person_height,
                frame_height=height,
            )

            for event in ppe_events.update(ppe_results):
                logger.warning(
                    "Возможное нарушение СИЗ: %s | камера: %s | track_id: %d",
                    event["event_type"],
                    camera_config["id"],
                    event["track_id"],
                )

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

            for event in fire_events.update(fire_detections):
                logger.warning(
                    "Событие: %s | камера: %s | confidence: %.2f | кадр: %d",
                    event["event_type"],
                    camera_config["id"],
                    event["confidence"],
                    frame_number,
                )

            draw_detections(frame, detections)
            draw_detections(frame, equipment)
            draw_detections(frame, fire_detections)
            for result in ppe_results:
                person = result["person"]

                helmet_text = (
                    "found" if result["has_helmet"] else "not found"
                )

                if result["has_vest"] is None:
                    vest_text = "unknown"
                else:
                    vest_text = (
                        "found" if result["has_vest"] else "not found"
                    )

                cv2.putText(
                    frame,
                    f"Helmet: {helmet_text} | Vest: {vest_text}",
                    (person.x1, min(height - 10, person.y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

            draw_zone(frame, detections, polygon_px)

            fps = fps_meter.tick()
            now = time.perf_counter()

            if now - last_fps_log >= 1.0:
                logger.info(
                    "FPS обработки: %.1f | "
                    "обработка моделями: %.1f мс | "
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
                f"AI: {inference_ms:.1f} ms | Objects: {len(detections)}",
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
