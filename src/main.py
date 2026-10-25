import logging
import time
from pathlib import Path

import yaml
import cv2

from src.capture.video_capture import VideoSource
from src.monitoring.fps import FPSMeter
from src.inference.detector import Detector
from src.inference.visualization import draw_detections

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
                detections = detector.predict(frame)
            except Exception:
                logger.exception("Ошибка обработки кадра детектором")
                raise

            inference_ms = (
                time.perf_counter() - inference_started
            ) * 1000

            draw_detections(frame, detections)

            fps = fps_meter.tick()
            now = time.perf_counter()

            if now - last_fps_log >= 1.0:
                logger.info(
                    "FPS обработки: %.1f | "
                    "вызов детектора: %.1f мс | "
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
                f"Detector: {inference_ms:.1f} ms | Objects: {len(detections)}",
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
