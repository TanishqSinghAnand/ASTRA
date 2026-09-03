"""Headless-safe display detection.

Discovered the hard way: on Linux with no X server at all, cv2.imshow /
cv2.namedWindow doesn't raise a catchable cv2.error — the underlying Qt
plugin loader aborts the whole process (SIGABRT). A try/except around it is
not enough. Check for a display *before* ever calling into the GUI, and
skip straight to headless behavior if there isn't one. Windows/macOS always
have a display for a foreground desktop app, so this only actually matters
on Linux (including this project's own dev sandbox, and any user running
the backend over SSH without X forwarding).
"""
from __future__ import annotations

import os
import sys


def gui_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True
