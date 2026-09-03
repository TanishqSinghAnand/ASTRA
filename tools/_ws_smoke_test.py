#!/usr/bin/env python3
"""One-off smoke test (not part of the pytest suite): connects to /ws/live
and prints every sequence_event until the experiment finishes or a timeout
elapses. Used to visually confirm the scripted mock demo scenario (correct,
correct, DEVIATION, recovery, correct, correct, complete) actually reaches
the browser-facing message layer end-to-end."""
import asyncio
import json
import sys
import time

import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8000/ws/live"
    deadline = time.time() + 30
    frame_msgs = 0
    async with websockets.connect(uri) as ws:
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if msg["type"] == "frame":
                frame_msgs += 1
                continue
            if msg["type"] == "sequence_event":
                print(f"[sequence_event] step={msg['step']} status={msg['status']} "
                      f"expected={msg['expected']} detected={msg['detected']} "
                      f"conf={msg['confidence']:.2f} recovered={msg.get('recovered')}")
                if msg["status"] == "COMPLETE":
                    print(f"Experiment finished. (frame messages received: {frame_msgs})")
                    return
            elif msg["type"] == "status":
                print(f"[status] {msg}")
    print("TIMEOUT waiting for experiment to complete", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
