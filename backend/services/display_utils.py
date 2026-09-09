"""Headless-safe display detection, and fitting frames to a display window.

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

import cv2
import numpy as np

# cv2.imshow renders whatever frame you pass it at 1:1 pixel scale — with a
# capture at HD (or higher) resolution, the window itself ends up bigger
# than the screen, cropping off overlays drawn near the frame's edges (e.g.
# zone boxes). fit_for_display() downscales only the *displayed* copy;
# detection/capture always runs on the original full-resolution frame
# first — quality is never traded away, only the on-screen rendering size.
DEFAULT_DISPLAY_MAX_WIDTH = 960


def fit_for_display(frame: np.ndarray, max_width: int = DEFAULT_DISPLAY_MAX_WIDTH) -> np.ndarray:
    if frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    new_size = (max_width, max(1, int(frame.shape[0] * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def gui_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True
