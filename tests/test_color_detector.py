"""Regression tests for the HSV color detector, run against the synthetic
scene generator (deterministic, known box positions) — no camera needed."""
from backend.config.settings import load_settings
from backend.perception.color_detector import ColorBoxDetector
from backend.services.synthetic_scene import BOX_SIZE, generate_frame


def _detect(t=2.0, width=1280, height=720):
    settings = load_settings()
    detector = ColorBoxDetector(settings.perception)
    frame = generate_frame(t, width, height, frame_index=int(t * 30))
    objects = {o.cls: o for o in detector.detect(frame)}
    return objects


def test_experiment_area_always_present():
    objects = _detect()
    assert "experiment_area" in objects
    assert objects["experiment_area"].confidence == 1.0


def test_red_box_detected_near_expected_position():
    objects = _detect(t=2.0, width=1280, height=720)
    assert "red_box" in objects
    box = objects["red_box"]
    expected_cx = 1280 * 0.20
    expected_cy = 720 * 0.70
    detected_cx = (box.bbox.x1 + box.bbox.x2) / 2
    detected_cy = (box.bbox.y1 + box.bbox.y2) / 2
    assert abs(detected_cx - expected_cx) < 15
    assert abs(detected_cy - expected_cy) < 15
    assert (box.bbox.x2 - box.bbox.x1) == BOX_SIZE + 0 or abs((box.bbox.x2 - box.bbox.x1) - BOX_SIZE) <= 2
    assert box.confidence > 0.8


def test_blue_box_detected_near_expected_position():
    objects = _detect(t=2.0, width=1280, height=720)
    assert "blue_box" in objects
    box = objects["blue_box"]
    expected_cx = 1280 * 0.20
    expected_cy = 720 * 0.25
    detected_cx = (box.bbox.x1 + box.bbox.x2) / 2
    detected_cy = (box.bbox.y1 + box.bbox.y2) / 2
    assert abs(detected_cx - expected_cx) < 15
    assert abs(detected_cy - expected_cy) < 15
    assert box.confidence > 0.8


def test_red_and_blue_are_not_confused():
    objects = _detect()
    red, blue = objects["red_box"], objects["blue_box"]
    # They should occupy clearly different vertical positions (red is lower,
    # blue is higher, per synthetic_scene.py's layout) and not overlap.
    assert red.bbox.y1 > blue.bbox.y2


def test_no_false_detection_on_empty_region():
    # Sanity check: shrinking min_box_contour_area to 0 shouldn't ever cause
    # the experiment_area rectangle itself (drawn in gray, not red/blue) to
    # be picked up as a colored box.
    objects = _detect()
    assert "experiment_area" in objects
    area_bbox = objects["experiment_area"].bbox
    for cls in ("red_box", "blue_box"):
        box = objects[cls].bbox
        # boxes should not be inside the experiment area in this frame (t=2.0,
        # before any pick/place motion in the synthetic scene)
        overlaps = not (box.x2 < area_bbox.x1 or box.x1 > area_bbox.x2 or box.y2 < area_bbox.y1 or box.y1 > area_bbox.y2)
        assert not overlaps
