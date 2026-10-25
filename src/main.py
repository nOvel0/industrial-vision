import logging
import time
from pathlib import Path
from collections import deque

import yaml
import cv2
import numpy as np

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
from src.events.engine import EventEngine
from src.storage.event_store import EventStore
from src.monitoring.logging_setup import setup_logging
from src.inference.spill_detector import SpillDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_NAME = "Industrial Vision"

logger = logging.getLogger(__name__)
camera_logger = logging.getLogger("camera_errors")
monitor_logger = logging.getLogger("system_monitoring")

def main() -> None:
    setup_logging(PROJECT_ROOT / "logs")

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
    camera_lost = False

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

    storage_path = Path(config["storage"]["root"])

    if not storage_path.is_absolute():
        storage_path = PROJECT_ROOT / storage_path

    event_store = EventStore(storage_path)

    event_engine = EventEngine(
        camera_id=camera_config["id"],
        store=event_store,
    )

    spill_config = config["spill"]

    spill_enabled = spill_config.get("enabled", False)

    spill_detector = SpillDetector(
        polygon=spill_config["polygon"],
        difference_threshold=int(spill_config["difference_threshold"]),
        min_area_ratio=float(spill_config["min_area_ratio"]),
    )

    spill_history = deque(
        maxlen=int(spill_config["confirmation_m"])
    )
    last_spill_event = None

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
                        camera_logger.error("%s", error)

                        if not camera_lost:
                            event_engine.emit({
                                "event_type": "camera_stream_lost",
                            })

                            camera_lost = True

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
                        spill_detector.reset()
                        spill_history.clear()
                        last_spill_event = None
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
                camera_logger.error("Видеопоток потерян: %s", error)

                if not camera_lost:
                    event_engine.emit({
                        "event_type": "camera_stream_lost",
                    })

                    camera_lost = True

                video.close()
                connected = False
                retry_at = time.perf_counter() + reconnect_sec
                continue

            if frame is None:
                logger.info("Чтение видеофайла завершено: конец файла или ошибка декодирования")
                break

            if camera_lost:
                event_engine.emit(
                    data={
                        "event_type": "camera_stream_restored",
                    },
                    frame=frame,
                )

                camera_logger.info("Получение кадров восстановлено")
                camera_lost = False

            evidence_frame = frame.copy()

            frame_number += 1

            inference_started = time.perf_counter()

            try:
                detections = tracker.update(frame)
                fire_detections = fire_detector.predict(frame)
                if spill_enabled:
                    spill_mask = spill_detector.predict(frame)
                else:
                    spill_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
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

            # разлив
            spill_present = cv2.countNonZero(spill_mask) > 0
            spill_history.append(spill_present)

            now = time.perf_counter()

            spill_confirmed = (
                len(spill_history) == spill_history.maxlen
                and sum(spill_history) >= int(spill_config["confirmation_n"])
            )

            spill_cooldown_finished = (
                last_spill_event is None
                or now - last_spill_event >= float(spill_config["cooldown_sec"])
            )

            if spill_present and spill_confirmed and spill_cooldown_finished:
                x, y, box_width, box_height = cv2.boundingRect(spill_mask)

                event_engine.emit(
                    data={
                        "event_type": "spill_detected",
                        "coordinates": {
                            "x": (x + box_width / 2) / width,
                            "y": (y + box_height / 2) / height,
                            "bbox": [
                                x / width,
                                y / height,
                                (x + box_width) / width,
                                (y + box_height) / height,
                            ],
                        },
                    },
                    frame=evidence_frame,
                )

                last_spill_event = now
            #

            ppe_results = check_ppe(
                people=detections,
                equipment=equipment,
                min_person_height=min_person_height,
                frame_height=height,
            )

            polygon_px = polygon_to_pixels(
                zone_polygon,
                width,
                height,
            )

            # Собираем подтверждённые события.
            pending_events = []
            pending_events.extend(ppe_events.update(ppe_results))
            pending_events.extend(zone_events.update(detections, polygon_px))
            pending_events.extend(fire_events.update(fire_detections))

            # Находим объект события по Track ID.
            objects_by_id = {
                obj.track_id: obj
                for obj in detections
                if obj.track_id is not None
            }

            for event in pending_events:
                obj = event.get("object")

                if obj is None:
                    obj = objects_by_id.get(event.get("track_id"))

                event_engine.emit(
                    data=event,
                    frame=evidence_frame,
                    obj=obj,
                )

            overlay = frame.copy()
            overlay[spill_mask > 0] = (255, 0, 255)

            cv2.addWeighted(
                overlay,
                0.35,
                frame,
                0.65,
                0,
                dst=frame,
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
                monitor_logger.info(
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

            key = cv2.waitKey(1)

            if key != -1:
                logger.info("Нажата клавиша: %d", key)

            if (key & 0xFF) in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        logger.info("Остановлено с клавиатуры")
    finally:
        video.close()
        cv2.destroyAllWindows()
        logger.info("Ресурсы освобождены")


if __name__ == "__main__":
    main()
