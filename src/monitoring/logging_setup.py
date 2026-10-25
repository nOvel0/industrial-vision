"""Настройка журналов с ограничением размера."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    def file_handler(filename: str) -> RotatingFileHandler:
        handler = RotatingFileHandler(
            logs_dir / filename,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        return handler

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            console,
            file_handler("application.log"),
        ],
        force=True,
    )

    for name, filename in (
        ("ai_events", "ai_events.log"),
        ("camera_errors", "camera_errors.log"),
        ("system_monitoring", "system_monitoring.log"),
    ):
        category_logger = logging.getLogger(name)
        category_logger.setLevel(logging.INFO)

        for handler in category_logger.handlers:
            handler.close()

        category_logger.handlers.clear()
        category_logger.addHandler(file_handler(filename))
        category_logger.propagate = True