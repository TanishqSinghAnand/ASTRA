"""WebSocket /ws/live — spec section 33. Streams `perception`,
`sequence_event`, and periodic annotated `frame` messages produced by
InferenceService._run_loop to every connected dashboard."""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("astra.ws")

router = APIRouter()


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    svc = websocket.app.state.inference_service
    await websocket.accept()
    queue = svc.subscribe()
    try:
        # Greet the client immediately with current status so the UI has
        # something to render before the first inference tick arrives.
        await websocket.send_json({"type": "status", **svc.status_dict()})
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        svc.unsubscribe(queue)
