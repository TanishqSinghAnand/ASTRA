"""Orchestrates one tick of CAMERA -> PERCEPTION -> ACTION -> SEQUENCE and
fans the results out to every connected WebSocket client. This is the one
place that wires the (currently mock) engines together — Phase 2-6 swap the
concrete PerceptionEngine/ActionRecognizer implementations assigned in
__init__ without changing anything below.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from backend.config.settings import Settings
from backend.experiment.experiment_loader import ExperimentDefinition
from backend.experiment.validator import RuleBasedSequenceEngine
from backend.mocks.mock_engines import MockActionRecognizer, MockPerceptionEngine
from backend.perception.base import ActionPrediction, PerceptionFrame, SequenceEvent
from backend.perception.interfaces import ActionRecognizer, PerceptionEngine, SequenceEngine
from backend.services.camera_service import CameraError, CameraService
from backend.services.frame_utils import encode_frame_jpeg_b64
from backend.services.logging_service import LoggingService
from backend.services.voice_service import VoiceService

logger = logging.getLogger("astra.inference")

FRAME_PUSH_INTERVAL_S = 0.15  # ~6-7 fps over the wire is plenty for a status view
LOOP_HZ = 15


class InferenceService:
    def __init__(self, settings: Settings, experiment: ExperimentDefinition, repo_root: Path):
        self.settings = settings
        self.experiment = experiment
        self.repo_root = repo_root

        self.camera = CameraService(settings.camera, repo_root)
        self.perception: PerceptionEngine = MockPerceptionEngine()
        self.action_recognizer: ActionRecognizer = MockActionRecognizer()
        self.sequence_engine: SequenceEngine = RuleBasedSequenceEngine(
            experiment, settings.perception.confidence_threshold
        )
        self.voice = VoiceService(experiment, enabled=settings.features.enable_voice)
        self.logging_service = LoggingService(experiment, settings.resolve_path(settings.paths.reports_dir))

        self.status: str = "IDLE"  # IDLE | RUNNING | FINISHED | STOPPED | ERROR
        self.error_message: str | None = None
        self.event_log: list[SequenceEvent] = []
        self.latest_perception: PerceptionFrame | None = None
        self.latest_prediction: ActionPrediction | None = None
        self.started_at: float | None = None

        self._subscribers: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None

    # -- pub/sub for the WebSocket layer -----------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _broadcast(self, message: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # Slow consumer: drop its oldest queued message rather than
                # blocking the whole inference loop on one client.
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self.status == "RUNNING":
            return
        try:
            self.camera.start()
        except CameraError as e:
            self.status = "ERROR"
            self.error_message = str(e)
            raise
        self.action_recognizer.reset()
        self.sequence_engine.reset()
        self.event_log.clear()
        self.error_message = None
        self.status = "RUNNING"
        self.started_at = time.time()
        self.voice.start()
        self.logging_service.start_run()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Inference loop started")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self.camera.stop()
        self.voice.stop()
        # No-op if the loop already finalized this run on completion — see
        # LoggingService.finalize_run's idempotency guard.
        self.logging_service.finalize_run(finished=self.sequence_engine.is_finished())
        if self.status == "RUNNING":
            self.status = "STOPPED"
        logger.info("Inference loop stopped")

    async def reset(self) -> None:
        await self.stop()
        self.action_recognizer.reset()
        self.sequence_engine.reset()
        self.event_log.clear()
        self.latest_perception = None
        self.latest_prediction = None
        self.error_message = None
        self.status = "IDLE"

    # -- status / snapshot for REST -----------------------------------------

    def status_dict(self) -> dict:
        step = self.experiment.step_by_id(self.sequence_engine.current_step())
        return {
            "status": self.status,
            "error": self.error_message,
            "camera_source": self.settings.camera.source,
            "camera_running": self.camera.is_running,
            "fps": round(self.camera.measured_fps, 1),
            "current_step": self.sequence_engine.current_step(),
            "total_steps": self.experiment.total_steps(),
            "current_instruction": step.instruction if step else None,
            "finished": self.sequence_engine.is_finished(),
            "device": "CPU (MediaPipe)",
            "offline": True,
        }

    # -- main loop ------------------------------------------------------------

    async def _run_loop(self) -> None:
        last_frame_push = 0.0
        try:
            while True:
                loop_start = time.time()
                captured = self.camera.read_latest()
                if captured is not None:
                    pframe = self.perception.process(captured.frame, captured.frame_index)
                    self.latest_perception = pframe
                    prediction = self.action_recognizer.update(pframe)
                    self.latest_prediction = prediction

                    await self._broadcast(
                        {
                            "type": "perception",
                            "action": prediction.action,
                            "confidence": prediction.confidence,
                            "step": self.sequence_engine.current_step(),
                            "fps": round(self.camera.measured_fps, 1),
                        }
                    )

                    # submit_action returns only its single "headline" event,
                    # but the auto-COMPLETE advance (validator.py) can append
                    # *two* events to history in one call — the real step's
                    # own CORRECT/RECOVERED plus the auto-generated COMPLETE.
                    # Diff history (concrete RuleBasedSequenceEngine state,
                    # not part of the abstract SequenceEngine interface) so
                    # every event actually gets logged/voiced/broadcast, not
                    # just the last one.
                    history_len_before = len(self.sequence_engine.history)
                    returned_event = self.sequence_engine.submit_action(prediction)
                    if returned_event is not None:
                        for event in self.sequence_engine.history[history_len_before:]:
                            self.event_log.append(event)
                            self.voice.on_sequence_event(event)
                            self.logging_service.log_event(event)
                            await self._broadcast({"type": "sequence_event", **event.model_dump()})
                        if self.sequence_engine.is_finished():
                            self.status = "FINISHED"
                            self.logging_service.finalize_run(finished=True)

                    if loop_start - last_frame_push >= FRAME_PUSH_INTERVAL_S:
                        last_frame_push = loop_start
                        image_b64 = encode_frame_jpeg_b64(captured.frame)
                        await self._broadcast(
                            {"type": "frame", "image": image_b64, "frame_index": captured.frame_index}
                        )

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.0, (1.0 / LOOP_HZ) - elapsed))
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive: never crash the app
            logger.exception("Inference loop crashed")
            self.status = "ERROR"
            self.error_message = "Inference loop encountered an unexpected error."
