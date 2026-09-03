#!/usr/bin/env python3
"""Capture the live sequence_event stream from /ws/live to a JSON file, for
building a replay/demo view of a real (mock-perception) backend run."""
import asyncio
import json
import time

import websockets


async def main():
    events = []
    t0 = None
    async with websockets.connect("ws://127.0.0.1:8000/ws/live") as ws:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if msg["type"] == "sequence_event":
                if t0 is None:
                    t0 = msg["timestamp"]
                msg["t_offset"] = round(msg["timestamp"] - t0, 2)
                events.append(msg)
                if msg["status"] == "COMPLETE":
                    break
    with open("data/reports/_demo_event_capture.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"Captured {len(events)} events")


if __name__ == "__main__":
    asyncio.run(main())
