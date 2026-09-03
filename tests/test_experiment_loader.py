import json

import pytest

from backend.experiment.experiment_loader import ExperimentLoadError, load_experiment


def test_load_valid_experiment(sample_experiment):
    assert sample_experiment.experiment_id == "BAS_SAMPLE_001"
    assert sample_experiment.total_steps() == 5
    assert sample_experiment.steps[0].action_key == "PICK_RED_BOX"
    assert sample_experiment.steps[1].action_key == "PLACE_RED_BOX"
    assert sample_experiment.steps[4].action_key == "COMPLETE_EXPERIMENT"


def test_step_by_id(sample_experiment):
    step = sample_experiment.step_by_id(3)
    assert step is not None
    assert step.action_key == "PICK_BLUE_BOX"
    assert sample_experiment.step_by_id(999) is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(ExperimentLoadError):
        load_experiment(tmp_path / "does_not_exist.json")


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ExperimentLoadError):
        load_experiment(bad)


def test_non_sequential_steps_raise(tmp_path):
    cfg = {
        "experiment_id": "BAD_001",
        "name": "Bad experiment",
        "steps": [
            {"id": 1, "action": "PICK", "object": "RED_BOX", "instruction": "x"},
            {"id": 3, "action": "PICK", "object": "BLUE_BOX", "instruction": "y"},
        ],
    }
    p = tmp_path / "bad_steps.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ExperimentLoadError):
        load_experiment(p)


def test_empty_steps_raise(tmp_path):
    cfg = {"experiment_id": "EMPTY_001", "name": "Empty", "steps": []}
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ExperimentLoadError):
        load_experiment(p)
