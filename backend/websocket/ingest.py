"""WebSocket /ws/ingest — the inverse of /ws/live: a client pushes its own
camera frames *in* (binary JPEG per message), for camera.source == "browser"
(see backend/config/settings.py, backend/services/browser_frame_source.py).
Exists for a cloud-hosted backend with no local capture device of its own —
the visiting browser's getUserMedia camera becomes the frame source instead.

Deliberately a separate endpoint from /ws/live rather than overloading it:
/ws/live's message shape is a fixed one-way server->client broadcast
contract every dashboard client already relies on; mixing in occasional
client->server binary frames on the same socket would complicate both
sides for no real benefit over just opening a second connection."""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("astra.ws.ingest")

router = APIRouter()


@router.websocket("/ws/ingest")
async def websocket_ingest(websocket: WebSocket) -> None:
    svc = websocket.app.state.inference_service
    await websocket.accept()
    if websocket.app.state.settings.camera.source != "browser":
        # Wrong deployment mode for this endpoint to do anything useful —
        # tell the client plainly instead of silently accepting frames
        # push_browser_frame() would just discard.
        await websocket.close(code=1008, reason="Backend is not configured for camera.source=browser")
        return
    try:
        while True:
            data = await websocket.receive_bytes()
            svc.push_browser_frame(data)
    except WebSocketDisconnect:
        logger.info("Browser camera source disconnected")
