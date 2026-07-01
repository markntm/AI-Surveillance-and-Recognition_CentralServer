from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os

from CC_data.database import engine, Base, SessionLocal
from CC_data.event_log import *
from CC_data.models import Behavior, Recognition, VehicleFunction, Event
from CC_data.schemas import EventIn, TelemetryIn, LiveTrackIn
from config.secret import dev_key, allowed_IP


app = FastAPI(title="Surveillance App", version="0.1")
templates = Jinja2Templates(directory="CC_dashboard/templates")
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_IP],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

API_KEY = os.getenv("SURVEILLANCE_API_KEY", dev_key)

_telemetry: dict = {}
_live_tracks: dict = {}


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"request": request})


@app.get("/events", include_in_schema=False, response_class=HTMLResponse)
def events(request: Request):
    return templates.TemplateResponse(request, "events.html", {"request": request})


@app.get("/cameras", include_in_schema=False, response_class=HTMLResponse)
def cameras(request: Request):
    return templates.TemplateResponse(request, "cameras.html", {"request": request})


@app.get("/analytics", include_in_schema=False, response_class=HTMLResponse)
def analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html", {"request": request})


@app.get("/api/status")
def get_status(db: Session = Depends(get_db)):
    """High-level system status for dashboard health checks."""

    last_event = (
        db.query(Event)
        .order_by(Event.timestamp.desc())
        .first()
    )

    total_events = db.query(Event).count()

    return {
        "server": "online",
        "total_events": total_events,
        "last_event_time": last_event.timestamp.isoformat() if last_event else None
    }


@app.get("/api/events")
def get_events(
        limit: int = 10,
        offset: int = 0,
        camera_id: Optional[str] = None,
        min_threat: Optional[int] = None,
        db: Session = Depends(get_db)
):
    """Retrieve recent surveillance events with detected objects and vehicles."""

    query = db.query(Event)

    # ---- Filtering ----
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)

    if min_threat is not None:
        query = query.filter(Event.threat_score >= min_threat)

    total_count = query.count()

    events = (
        query
        .order_by(Event.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    response = []

    for event in events:
        event_data = {
            "id": event.id,
            "camera_id": event.camera_id,
            "timestamp": event.timestamp.isoformat(),
            "threat_score": event.threat_score,
            "image_path": event.image_path,
            "objects": []
        }

        for obj in event.objects:
            obj_data = {
                "object_type": obj.object_type,
                "behavior": obj.behavior.name,
                "recognition": obj.recognition.name,
                "confidence": obj.confidence,
                "vehicle": None
            }

            if obj.vehicle:
                obj_data["vehicle"] = {
                    "license_plate": obj.vehicle.license_plate,
                    "primary_color": obj.vehicle.primary_color,
                    "vehicle_type": obj.vehicle.vehicle_type,
                    "vehicle_function": obj.vehicle.vehicle_function.name
                }

            event_data["objects"].append(obj_data)

        response.append(event_data)

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "events": response
    }


@app.get("/api/ingest/telemetry")
def get_telemetry():
    """Dashboard polls this to get the latest health stats per camera."""
    return list(_telemetry.values())


@app.get("/api/ingest/live")
def get_live_tracks(camera_id: Optional[str] = None):
    """Dashboard polls this to get the current live track list."""
    tracks = list(_live_tracks.values())
    if camera_id:
        tracks = [t for t in tracks if t["camera_id"] == camera_id]
    return tracks


@app.post("/api/events", dependencies=[Depends(verify_api_key)])
def ingest_event(event_in: EventIn, db: Session = Depends(get_db)):
    # 1. Create Event
    event = create_event(
        db=db,
        camera_id=event_in.camera_id,
        threat_score=event_in.threat_score,
        image_path=event_in.image_path
    )

    # 2. Add detected objects
    for obj in event_in.objects:
        detected = add_detected_object(
            db=db,
            event_id=event.id,
            object_type=obj.object_type,
            behavior=Behavior[obj.behavior],
            recognition=Recognition[obj.recognition],
            confidence=obj.confidence
        )

        # 3. Optional vehicle
        if obj.vehicle:
            add_vehicle(
                db=db,
                object_id=detected.id,
                license_plate=obj.vehicle.license_plate,
                primary_color=obj.vehicle.primary_color,
                vehicle_type=obj.vehicle.vehicle_type,
                vehicle_function=VehicleFunction[obj.vehicle.vehicle_function]
            )

    return {"status": "ok", "event_id": event.id}


@app.post("/api/ingest/telemetry", dependencies=[Depends(verify_api_key)])
def post_telemetry(data: TelemetryIn):
    """Receives health stats from cameras"""
    _telemetry[data.camera_id] = {
        "camera_id": data.camera_id,
        "workers_active": data.workers_active,
        "lpr_queue_size": data.lpr_queue_size,
        "active_tracks": data.active_tracks,
        "last_seen": datetime.utcnow().isoformat()
    }
    return {"status": "ok"}


@app.post("/api/ingest/live", dependencies=[Depends(verify_api_key)])
def post_live(data: LiveTrackIn):
    """Receives per-track live updates from the camera client."""
    _live_tracks[data.track_id] = {
        "camera_id": data.camera_id,
        "track_id": data.track_id,
        "label": data.label,
        "confidence": data.confidence,
        "license_plate": data.license_plate,
        "last_seen": datetime.utcnow().isoformat()
    }
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
