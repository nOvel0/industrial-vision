"""Общий формат события системы."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Event:
    event_type: str
    camera_id: str
    confidence: float | None = None
    track_id: int | None = None
    zone_id: str | None = None
    coordinates: dict = field(default_factory=dict)
    snapshot: str | None = None
    status: str = "new"

    event_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )