"""ASTRA backend entrypoint.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

(from the repo root, with the venv activated).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router
from backend.config.settings import REPO_ROOT, get_settings
from backend.experiment.experiment_loader import ExperimentLoadError, load_experiment
from backend.services.inference_service import InferenceService
from backend.websocket.ingest import router as ws_ingest_router
from backend.websocket.live import router as ws_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("astra.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    experiment_path = settings.resolve_path(settings.paths.experiment_config)
    try:
        experiment = load_experiment(experiment_path)
    except ExperimentLoadError as e:
        # Fail loudly at startup rather than crashing later mid-demo.
        logger.error("Failed to load experiment config: %s", e)
        raise

    app.state.settings = settings
    app.state.experiment = experiment
    app.state.inference_service = InferenceService(settings, experiment, REPO_ROOT)
    logger.info(
        "ASTRA backend ready. Experiment=%s (%d steps). Camera source=%s.",
        experiment.experiment_id,
        experiment.total_steps(),
        settings.camera.source,
    )
    yield
    await app.state.inference_service.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ASTRA",
        description="Autonomous Space Task Recognition & Assistance — SIH26174 prototype backend.",
        version="0.1.0-mvp",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local prototype only — tighten if ever exposed beyond localhost
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(ws_router)
    app.include_router(ws_ingest_router)

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "ASTRA",
            "description": "Autonomous Space Task Recognition & Assistance",
            "status": "offline-ai prototype backend running",
            "docs": "/docs",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("backend.main:app", host=settings.server.host, port=settings.server.port, reload=False)
