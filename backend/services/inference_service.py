"""Orchestrates one tick of CAMERA -> PERCEPTION -> ACTION -> SEQUENCE and
fans the results out to every connected WebSocket client. This is the one
place that wires the engines together: real Phase 2/4 perception+action by
default, with the scripted Phase 0/1 mocks available as an opt-in fallback
(features.use_mock_engines) for headless/CI verification where no real
person is in frame to produce anything meaningful.
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
from backend.perception.real_engine import RealPerceptionEngine
from backend.services.camera_service import CameraError, CameraService
from backend.services.frame_utils import encode_frame_jpeg_b64
from backend.services.logging_service import LoggingService
from backend.services.overlay import draw_perception_overlay
from backend.services.recording_service import RecordingService
from backend.services.voice_service import VoiceService
from backend.temporal.action_recognizer import RuleBasedActionRecognizer

logger = logging.getLogger("astra.inference")

LOOP_HZ = 15  # inference loop's target cadence — actual rate is capped by
              # however long perception.process() takes on real hardware
STREAM_HZ = 20  # video stream's own cadence — decoupled from inference
                # speed (see _stream_loop's docstring for why this matters)
STREAM_MAX_WIDTH = 640  # downscale before encoding; a live status view
                         # doesn't need the camera's full capture resolution


class InferenceService:
    def __init__(self, settings: Settings, experiment: ExperimentDefinition, repo_root: Path):
        self.settings = settings
        self.experiment = experiment
        self.repo_root = repo_root

        self.camera = CameraService(settings.camera, repo_root)
        if settings.features.use_mock_engines:
            self.perception: PerceptionEngine = MockPerceptionEngine()
            self.action_recognizer: ActionRecognizer = MockActionRecognizer()
        else:
            self.perception = RealPerceptionEngine(settings.perception)
            self.action_recognizer = RuleBasedActionRecognizer(settings.perception, settings.temporal)
        self.sequence_engine: SequenceEngine = RuleBasedSequenceEngine(
            experiment, settings.perception.confidence_threshold
        )
        self.voice = VoiceService(experiment, enabled=settings.features.enable_voice)
        self.logging_service = LoggingService(experiment, settings.resolve_path(settings.paths.reports_dir))
        self.recording = RecordingService(
            settings.resolve_path(settings.paths.recordings_dir),
            enabled=settings.features.enable_recording,
            nominal_fps=settings.camera.target_fps,
        )

        self.status: str = "IDLE"  # IDLE | RUNNING | FINISHED | STOPPED | ERROR
        self.error_message: str | None = None
        self.event_log: list[SequenceEvent] = []
        self.latest_perception: PerceptionFrame | None = None
        self.latest_prediction: ActionPrediction | None = None
        self.started_at: float | None = None

        self._subscribers: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None
        self._stream_task: asyncio.Task | None = None

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
        self.recording.start_run()
        self._task = asyncio.create_task(self._inference_loop())
        self._stream_task = asyncio.create_task(self._stream_loop())
        logger.info("Inference loop started")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._stream_task is not None:
            self._stream_task.cancel()
            self._stream_task = None
        self.camera.stop()
        self.voice.stop()
        self.recording.finalize_run()
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

    # -- main loops ------------------------------------------------------------
    #
    # Two independent loops, not one: perception.process() is a real,
    # CPU-bound ML call (MediaPipe pose+hands, HSV or — v2.0 — YOLO object
    # detection) that can take anywhere from ~50ms to 500ms+ depending on
    # machine load. Doing everything in one loop meant the *video feed*
    # itself was gated by inference speed — under load the dashboard's
    # camera view would visibly stutter/lag, because a slow detection frame
    # blocked the next frame push too. Splitting them means the raw video
    # stays smooth at its own cadence regardless of how fast detection is
    # keeping up; the annotation overlay just uses whatever perception
    # result is most recently available (self.latest_perception), which
    # under heavy load may trail the video by a frame or two — imperceptible
    # in practice, and far better than the whole feed freezing.

    async def _inference_loop(self) -> None:
        try:
            while True:
                loop_start = time.time()
                captured = self.camera.read_latest()
                if captured is not None:
                    # Off the event loop thread: this is the one truly
                    # expensive call in either loop, and without to_thread
                    # it would block *all* asyncio activity (including the
                    # stream loop below, and any other WS/HTTP traffic) for
                    # its entire duration, not just this loop's own pacing.
                    pframe = await asyncio.to_thread(self.perception.process, captured.frame, captured.frame_index)
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

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.0, (1.0 / LOOP_HZ) - elapsed))
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive: never crash the app
            logger.exception("Inference loop crashed")
            self.status = "ERROR"
            self.error_message = "Inference loop encountered an unexpected error."

    async def _stream_loop(self) -> None:
        try:
            while True:
                loop_start = time.time()
                captured = self.camera.read_latest()
                if captured is not None:
                    pframe = self.latest_perception
                    annotated = draw_perception_overlay(captured.frame, pframe) if pframe is not None else captured.frame
                    self.recording.write_frame(annotated)
                    image_b64 = encode_frame_jpeg_b64(annotated, max_width=STREAM_MAX_WIDTH)
                    await self._broadcast(
                        {"type": "frame", "image": image_b64, "frame_index": captured.frame_index}
                    )

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.0, (1.0 / STREAM_HZ) - elapsed))
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive: never crash the app
            logger.exception("Stream loop crashed")
            self.status = "ERROR"
            self.error_message = "Video stream loop encountered an unexpected error."
