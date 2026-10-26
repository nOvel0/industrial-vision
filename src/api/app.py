"""REST API: состояние, события и изменение настроек."""

import json
from pathlib import Path
from queue import SimpleQueue
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="Industrial Vision API")

storage_root = Path("events")
commands = SimpleQueue()

state = {
    "camera_id": "cam_01",
    "camera": "starting",
    "fps": 0.0,
}


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: float = Field(gt=0, le=1)
    ppe: float = Field(gt=0, le=1)
    fire_smoke: float = Field(gt=0, le=1)


Coordinate = Annotated[float, Field(ge=0, le=1)]


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    polygon: list[tuple[Coordinate, Coordinate]] = Field(min_length=3)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return state.copy()


@app.get("/cameras")
def cameras():
    return [{
        "camera_id": state["camera_id"],
        "status": state["camera"],
    }]


@app.get("/events")
def events(limit: int = Query(default=50, ge=1, le=100)):
    paths = storage_root.glob("*/*/*/*/event.json")
    result = []

    for path in paths:
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
            result.append(event)
        except (OSError, json.JSONDecodeError):
            continue

    result.sort(key=lambda item: item["timestamp"], reverse=True)

    return result[:limit]


@app.get("/events/{event_id}")
def event_by_id(event_id: UUID):
    for path in storage_root.glob(f"*/*/*/{event_id}/event.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

    raise HTTPException(status_code=404, detail="Event not found")


@app.post("/config/thresholds", status_code=202)
def update_thresholds(payload: Thresholds):
    values = payload.model_dump()

    commands.put(("thresholds", values))

    return {
        "status": "queued",
        "thresholds": values,
    }


@app.post("/config/zones", status_code=202)
def update_zones(payload: Zone):
    commands.put(("zone", payload.polygon))

    return {
        "status": "queued",
        "polygon": payload.polygon,
    }