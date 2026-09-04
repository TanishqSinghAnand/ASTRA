"""Throwaway manual smoke test for Phase 7 (backend/services/voice_service.py)
against real hardware (SAPI5 on Windows): confirms pyttsx3 actually
initializes and speaks without hanging or crashing. Unit tests use a fake
engine and can't catch a real driver/hardware problem.

    python tools/_smoke_voice.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import REPO_ROOT  # noqa: E402
from backend.experiment.experiment_loader import load_experiment  # noqa: E402
from backend.services.voice_service import VoiceService  # noqa: E402

experiment = load_experiment(REPO_ROOT / "experiments" / "bas_sample_001.json")
service = VoiceService(experiment, enabled=True)

print("Starting voice service and speaking a test line...")
service.start()
service.speak("A S T R A voice guidance test. If you can hear this, Phase 7 works.")
time.sleep(0.5)  # let the worker thread pick the item up before we start stopping
t0 = time.time()
service.stop()
elapsed = time.time() - t0
print(f"stop() returned after {elapsed:.2f}s (should be roughly the time it took to finish speaking).")
print("Done.")
