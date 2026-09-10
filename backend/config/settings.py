"""Configuration loading for ASTRA.

Loads config/config.yaml, applies environment-variable overrides (so a demo
laptop can flip a flag without editing the file), and exposes a single
validated Settings object the rest of the backend imports.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (backend/config/settings.py -> astra/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class CameraSettings(BaseModel):
    # "browser": no local capture device at all — frames arrive pushed over
    # /ws/ingest from a client's own getUserMedia camera (see
    # backend/services/browser_frame_source.py). For a cloud deployment
    # with no camera attached to the server itself, letting the visiting
    # browser supply its own webcam frames instead of the backend reading
    # a local device.
    source: Literal["webcam", "video_file", "synthetic", "browser"] = "synthetic"
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    video_file_path: str = "data/recordings/demo_clip.mp4"


class HSVRange(BaseModel):
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


class ColorSpec(BaseModel):
    ranges: list[HSVRange] = Field(default_factory=list)


class PerceptionSettings(BaseModel):
    confidence_threshold: float = 0.60
    # Distance (px, at a 640px-wide reference frame — see
    # backend/perception/interaction.py's _THRESHOLD_REFERENCE_FRAME_WIDTH)
    # a hand must stay within, continuously, to confirm a PICK — see
    # pick_dwell_seconds below. Scaled to actual capture resolution, same
    # as the old hand_object_touch_distance_px this replaces.
    hand_touch_distance_px: float = 150.0
    # How many continuous seconds a hand must stay within
    # hand_touch_distance_px of an object before it counts as picked up.
    # Wall-clock seconds, not a frame count, so behavior is the same
    # regardless of the machine's actual frame rate (a slower laptop
    # shouldn't need a shorter "hold still" gesture than a faster one).
    # Requiring sustained contact — not just one close frame — filters out
    # a hand briefly passing near the object without truly gripping it.
    pick_dwell_seconds: float = 3.0
    # How many continuous seconds an object's position must stay inside a
    # configured zone before it counts as placed there. Checked purely
    # from the object's own detected position — a hand lingering in or
    # near the zone during this window doesn't reset it either way.
    place_dwell_seconds: float = 3.0
    min_box_contour_area: int = 800
    experiment_area: tuple[float, float, float, float] = (0.55, 0.55, 0.85, 0.85)
    colors: dict[str, ColorSpec] = Field(default_factory=dict)

    # --- v2.0: general object detection (backend/perception/object_detector.py) ---
    # "hsv" keeps v1's color-marker approach available; "yolo" is v2.0's
    # default — real object identity (phone, cup, bottle, ...), no markers.
    detector_backend: Literal["hsv", "yolo"] = "yolo"
    yolo_model: str = "yolov8n.pt"
    yolo_classes: list[str] = Field(default_factory=list)  # COCO class names to track, e.g. ["cell phone", "cup"]
    yolo_confidence_threshold: float = 0.35
    # YOLO's own internal inference resolution (its default is 640). Lower
    # is meaningfully faster on CPU (~(640/imgsz)^2 fewer pixels processed)
    # at some accuracy cost — 320 is plenty for a close-up tabletop demo
    # where tracked objects fill a decent fraction of the frame; raise it
    # if detection misses small/distant objects.
    yolo_imgsz: int = 320

    # --- v2.0: multiple named placement zones, not just one ---
    # Each maps to a fraction-of-frame bbox [x1, y1, x2, y2], same shape as
    # v1's single `experiment_area`. ExperimentStep.target names which zone
    # (by this dict's key, case-insensitive) an object must land in.
    target_zones: dict[str, tuple[float, float, float, float]] = Field(default_factory=dict)


class TemporalSettings(BaseModel):
    action_stability_frames: int = 6
    min_confidence_duration_ms: int = 300


class FeatureFlags(BaseModel):
    enable_voice: bool = True
    enable_recording: bool = True
    enable_streaming: bool = False
    demo_mode: bool = True
    # Opt-in fallback to the scripted Phase 0/1 mocks instead of the real
    # Phase 2/4 perception+action pipeline — for headless/CI verification
    # where no real person is in frame to produce anything meaningful.
    use_mock_engines: bool = False


class PathSettings(BaseModel):
    experiment_config: str = "experiments/bas_sample_001.json"
    recordings_dir: str = "data/recordings"
    reports_dir: str = "data/reports"
    raw_data_dir: str = "data/raw"


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}

    camera: CameraSettings = Field(default_factory=CameraSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    paths: PathSettings = Field(default_factory=PathSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path from config relative to the repo root."""
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p


_ENV_OVERRIDES = {
    "CAMERA_INDEX": ("camera", "camera_index", int),
    "CAMERA_SOURCE": ("camera", "source", str),
    "FRAME_WIDTH": ("camera", "frame_width", int),
    "FRAME_HEIGHT": ("camera", "frame_height", int),
    "CONFIDENCE_THRESHOLD": ("perception", "confidence_threshold", float),
    "ACTION_STABILITY_FRAMES": ("temporal", "action_stability_frames", int),
    "ENABLE_VOICE": ("features", "enable_voice", lambda v: v.lower() == "true"),
    "ENABLE_RECORDING": ("features", "enable_recording", lambda v: v.lower() == "true"),
    "ENABLE_STREAMING": ("features", "enable_streaming", lambda v: v.lower() == "true"),
    "DEMO_MODE": ("features", "demo_mode", lambda v: v.lower() == "true"),
    "USE_MOCK_ENGINES": ("features", "use_mock_engines", lambda v: v.lower() == "true"),
    "MODEL_DEVICE": ("model_device", None, str),
}


def _apply_env_overrides(raw: dict) -> dict:
    for env_key, (section, field, caster) in _ENV_OVERRIDES.items():
        if env_key not in os.environ:
            continue
        value = caster(os.environ[env_key])
        if field is None:
            raw[section] = value
        else:
            raw.setdefault(section, {})
            raw[section][field] = value
    return raw


def load_settings(config_path: Path | str | None = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    raw = _apply_env_overrides(raw)
    return Settings(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton accessor — import this everywhere else in the backend."""
    return load_settings()
