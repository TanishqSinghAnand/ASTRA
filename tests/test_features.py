"""Tests for training/features.py. Same hand-constructed-PerceptionFrame
approach as test_interaction.py/test_action_recognizer.py."""
from __future__ import annotations

from backend.perception.base import BBox, DetectedObject, HandFrame, Landmark, PerceptionFrame
from training.features import MISSING, extract_window_features

FRAME_W, FRAME_H = 640, 480


def _box(center: tuple[float, float], cls: str = "red_box", confidence: float = 0.9, half_size: float = 20.0) -> DetectedObject:
    cx, cy = center
    return DetectedObject(
        **{"class": cls},
        confidence=confidence,
        bbox=BBox(x1=cx - half_size, y1=cy - half_size, x2=cx + half_size, y2=cy + half_size),
    )


def _hand(center_px: tuple[float, float]) -> HandFrame:
    nx, ny = center_px[0] / FRAME_W, center_px[1] / FRAME_H
    return HandFrame(
        detected=True,
        landmarks=[
            Landmark(name="wrist", x=nx, y=ny),
            Landmark(name="index_mcp", x=nx, y=ny),
            Landmark(name="pinky_mcp", x=nx, y=ny),
        ],
    )


def _frame(
    frame_index: int,
    timestamp: float,
    box_center: tuple[float, float] | None,
    hand_center_px_: tuple[float, float] | None,
    box_confidence: float = 0.9,
) -> PerceptionFrame:
    objects = [_box(box_center, confidence=box_confidence)] if box_center is not None else []
    hands = [_hand(hand_center_px_)] if hand_center_px_ is not None else []
    return PerceptionFrame(
        frame_index=frame_index,
        timestamp=timestamp,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        objects=objects,
        hands=hands,
    )


def test_empty_window_returns_empty_dict():
    assert extract_window_features([], ["red_box"]) == {}


def test_frames_with_unknown_dimensions_are_excluded():
    frame = PerceptionFrame(frame_index=0, frame_width=0, frame_height=0)
    assert extract_window_features([frame], ["red_box"]) == {}


def test_single_frame_window_has_no_velocity_but_has_distance_and_confidence():
    frame = _frame(0, 0.0, box_center=(100, 100), hand_center_px_=(110, 100), box_confidence=0.8)
    features = extract_window_features([frame], ["red_box"])

    assert features["red_box_displacement_px"] == MISSING
    assert features["red_box_velocity_px_per_s"] == MISSING
    assert features["red_box_mean_hand_distance_px"] == 10.0
    assert features["red_box_min_hand_distance_px"] == 10.0
    assert features["red_box_mean_confidence"] == 0.8


def test_class_never_detected_is_all_missing():
    frame = _frame(0, 0.0, box_center=(100, 100), hand_center_px_=(110, 100))
    features = extract_window_features([frame], ["blue_box"])

    assert features["blue_box_mean_hand_distance_px"] == MISSING
    assert features["blue_box_min_hand_distance_px"] == MISSING
    assert features["blue_box_mean_confidence"] == MISSING
    assert features["blue_box_displacement_px"] == MISSING
    assert features["blue_box_velocity_px_per_s"] == MISSING


def test_multi_frame_window_computes_displacement_and_velocity():
    frames = [
        _frame(0, 0.0, box_center=(100, 100), hand_center_px_=(110, 100)),
        _frame(1, 1.0, box_center=(200, 100), hand_center_px_=(210, 100)),
    ]
    features = extract_window_features(frames, ["red_box"])

    assert features["red_box_displacement_px"] == 100.0
    assert features["red_box_velocity_px_per_s"] == 100.0  # 100px over 1.0s


def test_hand_distance_uses_nearest_hand_when_multiple_present():
    frame = PerceptionFrame(
        frame_index=0,
        timestamp=0.0,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        objects=[_box((100, 100))],
        hands=[_hand((500, 400)), _hand((110, 100))],  # far hand, near hand
    )
    features = extract_window_features([frame], ["red_box"])
    assert features["red_box_min_hand_distance_px"] == 10.0


def test_frame_with_no_hands_has_missing_distance_but_present_confidence():
    frame = _frame(0, 0.0, box_center=(100, 100), hand_center_px_=None, box_confidence=0.7)
    features = extract_window_features([frame], ["red_box"])
    assert features["red_box_mean_hand_distance_px"] == MISSING
    assert features["red_box_mean_confidence"] == 0.7


def test_multiple_tracked_classes_are_independent():
    frame = PerceptionFrame(
        frame_index=0,
        timestamp=0.0,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        objects=[_box((100, 100), cls="red_box"), _box((400, 300), cls="blue_box")],
        hands=[_hand((110, 100))],
    )
    features = extract_window_features([frame], ["red_box", "blue_box"])
    assert features["red_box_mean_hand_distance_px"] == 10.0
    assert features["blue_box_mean_hand_distance_px"] != MISSING  # far hand still counted as nearest
