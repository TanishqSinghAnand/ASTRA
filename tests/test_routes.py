"""Tests for backend/api/routes.py's /api/mode endpoint — the dashboard's
two-tab (color detector vs YOLO detector) switcher. Sets
USE_MOCK_ENGINES=true before importing the app so this exercises the
mode-switching wiring itself (which config gets loaded, which experiment
gets attached, that the previous service's camera/loops actually stop)
without needing a real camera or a MediaPipe model load.
"""
from __future__ import annotations

import os

os.environ["USE_MOCK_ENGINES"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


def test_get_mode_defaults_to_color():
    with TestClient(app) as client:
        resp = client.get("/api/mode")
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "color"
        assert set(body["available"]) == {"color", "yolo"}


def test_switch_to_yolo_swaps_experiment():
    with TestClient(app) as client:
        resp = client.post("/api/mode/yolo")
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "yolo"
        object_ids = set(body["experiment"]["objects"].keys())
        assert {"MOUSE", "BOTTLE"} & object_ids

        # get_mode reflects the switch too.
        assert client.get("/api/mode").json()["mode"] == "yolo"

        # A fresh InferenceService was built for the new mode, not the
        # previous one left running.
        assert body["status"]["status"] == "IDLE"


def test_switch_back_to_color_restores_original_experiment():
    with TestClient(app) as client:
        client.post("/api/mode/yolo")
        resp = client.post("/api/mode/color")
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "color"
        object_ids = set(body["experiment"]["objects"].keys())
        assert {"BLUE_BOX", "YELLOW_BOX"} <= object_ids


def test_unknown_mode_is_rejected():
    with TestClient(app) as client:
        resp = client.post("/api/mode/bogus")
        assert resp.status_code == 400
        # Rejecting an unknown mode must not have torn down the running
        # service — still on the original mode afterward.
        assert client.get("/api/mode").json()["mode"] == "color"


def test_mode_switch_does_not_orphan_existing_subscribers():
    """Regression: /ws/live captures whichever InferenceService instance
    is current *at connect time* and holds that reference for the
    socket's entire lifetime (backend/websocket/live.py). Swapping in a
    brand-new InferenceService on mode switch — instead of reconfiguring
    the existing one in place — would silently orphan every already-
    connected dashboard: still "connected", but subscribed to a dead
    object's queue that nothing ever broadcasts to again. Observed
    directly: switching tabs left the video feed stuck on "WAITING FOR
    FIRST FRAME..." forever, no error, no reconnect, because the socket
    never itself failed."""
    with TestClient(app) as client:
        svc_before = app.state.inference_service
        queue = svc_before.subscribe()

        resp = client.post("/api/mode/yolo")
        assert resp.status_code == 200

        # Reconfigured in place -- not replaced -- so the pre-switch
        # subscription is still registered against the live instance.
        assert app.state.inference_service is svc_before
        assert queue in svc_before._subscribers
