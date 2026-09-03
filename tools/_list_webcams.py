"""Probe camera index 0 at several requested resolutions (DirectShow backend)
to find its actual native/max HD capability, since OpenCV opens at a low
default (640x480) unless a higher mode is explicitly requested.
"""
import cv2
import numpy as np

candidates = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

idx = 0
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"index {idx}: NOT AVAILABLE")
else:
    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        if ok and frame is not None:
            std = float(np.std(frame))
            print(f"requested {w}x{h} -> actual {got_w}x{got_h}  frame_shape={frame.shape}  frame_std={std:.1f}")
        else:
            print(f"requested {w}x{h} -> actual {got_w}x{got_h}  NO FRAME")
    # Save a frame at the highest mode that worked
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ok, frame = cap.read()
    if ok and frame is not None:
        cv2.imwrite("data/recordings/_probe_index_0_hd.png", frame)
        print("saved -> data/recordings/_probe_index_0_hd.png")
    cap.release()
