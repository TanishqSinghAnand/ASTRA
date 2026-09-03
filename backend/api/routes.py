"""REST API — spec section 33. Every endpoint is a thin wrapper around the
shared InferenceService instance stored on app.state; no business logic
lives here."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.services.inference_service import InferenceService

router = APIRouter(prefix="/api")


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
