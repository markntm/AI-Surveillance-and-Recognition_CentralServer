from pydantic import BaseModel
from typing import List, Optional


class VehicleIn(BaseModel):
    license_plate: str
    primary_color: str
    vehicle_type: str
    vehicle_function: str


class DetectedObjectIn(BaseModel):
    object_type: str
    behavior: str
    recognition: str
    confidence: float
    vehicle: Optional[VehicleIn] = None


class EventIn(BaseModel):
    camera_id: str
    threat_score: int
    image_path: Optional[str] = None
    objects: List[DetectedObjectIn]


class TelemetryIn(BaseModel):
    camera_id: str
    workers_active: int
    lpr_queue_size: int
    active_tracks: int


class LiveTrackIn(BaseModel):
    camera_id: str
    track_id: str
    label: str
    confidence: float
    license_plate: Optional[str] = None
