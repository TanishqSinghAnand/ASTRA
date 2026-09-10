"""REST API — spec section 33. Every endpoint is a thin wrapper around the
shared InferenceService instance stored on app.state; no business logic
lives here."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.config.settings import REPO_ROOT, load_settings
from backend.experiment.experiment_loader import ExperimentLoadError, load_experiment
from backend.services.inference_service import InferenceService

router = APIRouter(prefix="/api")

# Two tabs in the dashboard, two detector backends — each needs its own
# config.yaml (different detector_backend/colors/yolo_classes *and*
# experiment_config, since the two use different tracked object ids: the
# color path's BLUE_BOX/YELLOW_BOX vs YOLO's CUP/BOTTLE). Only one runs at
# a time — a webcam can't be opened by two capture handles at once, and
# running both detectors' models simultaneously on CPU for a "compare
# them side by side" view was never the ask — so switching modes tears
# down and rebuilds the whole InferenceService rather than keeping two
# alive.
_MODE_CONFIGS = {
    "color": REPO_ROOT / "config" / "config.yaml",
    "yolo": REPO_ROOT / "config" / "config.yolo.yaml",
}


def _svc(request: Request) -> InferenceService:
    return request.app.state.inference_service


@router.get("/status")
async def get_status(request: Request) -> dict:
    return _svc(request).status_dict()


@router.get("/experiment")
async def get_experiment(request: Request) -> dict:
    return _svc(request).experiment.model_dump()


@router.post("/experiment/start")
async def start_experiment(request: Request) -> dict:
    svc = _svc(request)
    try:
        await svc.start()
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return svc.status_dict()


@router.post("/experiment/stop")
async def stop_experiment(request: Request) -> dict:
    svc = _svc(request)
    await svc.stop()
    return svc.status_dict()


@router.post("/experiment/reset")
async def reset_experiment(request: Request) -> dict:
    svc = _svc(request)
    await svc.reset()
    return svc.status_dict()


@router.get("/experiment/state")
async def get_state(request: Request) -> dict:
    return _svc(request).status_dict()


@router.get("/experiment/log")
async def get_log(request: Request) -> dict:
    svc = _svc(request)
    return {"events": [e.model_dump() for e in svc.event_log]}


@router.get("/mode")
async def get_mode(request: Request) -> dict:
    return {"mode": getattr(request.app.state, "mode", "color"), "available": list(_MODE_CONFIGS)}


@router.post("/mode/{mode}")
async def set_mode(mode: str, request: Request) -> dict:
    """Stops the current run and reconfigures the *same* InferenceService
    in place from the other detector's config.yaml (InferenceService.
    reconfigure's docstring explains why: swapping in a whole new instance
    would silently orphan every already-connected dashboard's WebSocket
    subscription). The dashboard should treat the response's `experiment`
    as a full replacement (different object ids/steps, not just a status
    refresh)."""
    if mode not in _MODE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown mode {mode!r}; expected one of {list(_MODE_CONFIGS)}")
    app = request.app
    svc = _svc(request)
    await svc.stop()

    settings = load_settings(_MODE_CONFIGS[mode])
    try:
        experiment = load_experiment(settings.resolve_path(settings.paths.experiment_config))
    except ExperimentLoadError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load experiment for mode {mode!r}: {e}") from e

    svc.reconfigure(settings, experiment)
    app.state.settings = settings
    app.state.experiment = experiment
    app.state.mode = mode
    return {
        "mode": mode,
        "status": svc.status_dict(),
        "experiment": experiment.model_dump(),
    }
