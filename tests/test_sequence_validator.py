"""Unit tests for the sequence/decision logic — spec section 43 test names.
These exercise backend/experiment/validator.py directly with synthetic
ActionPrediction objects, so they need no camera, no perception model, and
run in well under a second."""
from backend.experiment.validator import RuleBasedSequenceEngine
from backend.perception.base import ActionPrediction, SequenceStatus


def predict(action, confidence=0.9):
    return ActionPrediction(action=action, confidence=confidence)


def test_correct_sequence(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    sequence = [
        "PICK_RED_BOX",
        "PLACE_RED_BOX",
        "PICK_BLUE_BOX",
        "PLACE_BLUE_BOX",
        "COMPLETE_EXPERIMENT",
    ]
    statuses = []
    for action in sequence:
        event = engine.submit_action(predict(action))
        assert event is not None
        statuses.append(event.status)
    assert statuses[:-1] == [SequenceStatus.CORRECT] * 4
    assert statuses[-1] == SequenceStatus.COMPLETE
    assert engine.is_finished()


def test_wrong_action(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    engine.submit_action(predict("PICK_RED_BOX"))  # step 1, correct
    # Step 2 expects PLACE_RED_BOX; deliberately do something else instead.
    event = engine.submit_action(predict("PICK_BLUE_BOX"))
    assert event.status == SequenceStatus.OUT_OF_SEQUENCE
    assert event.expected == "PLACE_RED_BOX"
    assert event.detected == "PICK_BLUE_BOX"
    assert engine.current_step() == 2  # did not advance


def test_skipped_step(sample_experiment):
    # 1 -> 3 (step 2 never happened). The MVP validator classifies any
    # detected action that doesn't match the current expected step and
    # hasn't already been completed as OUT_OF_SEQUENCE; a dedicated
    # SKIPPED_STEP classification (distinguishing "never happened" from
    # "happened later") is added with the full state machine in Phase 5/6.
    engine = RuleBasedSequenceEngine(sample_experiment)
    engine.submit_action(predict("PICK_RED_BOX"))  # step 1, correct
    event = engine.submit_action(predict("PICK_BLUE_BOX"))  # skips step 2 (PLACE_RED_BOX)
    assert event.status == SequenceStatus.OUT_OF_SEQUENCE
    assert engine.current_step() == 2


def test_recovery(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    engine.submit_action(predict("PICK_RED_BOX"))
    error_event = engine.submit_action(predict("PICK_BLUE_BOX"))
    assert error_event.status == SequenceStatus.OUT_OF_SEQUENCE

    recovery_event = engine.submit_action(predict("PLACE_RED_BOX"))
    assert recovery_event.status == SequenceStatus.RECOVERED
    assert recovery_event.recovered is True
    assert engine.current_step() == 3

    # The step *after* a recovery is a normal correct step again, not another recovery.
    next_event = engine.submit_action(predict("PICK_BLUE_BOX"))
    assert next_event.status == SequenceStatus.CORRECT


def test_repeated_step(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    engine.submit_action(predict("PICK_RED_BOX"))  # step 1 done
    # Now at step 2 (PLACE_RED_BOX); repeat step 1's action instead.
    event = engine.submit_action(predict("PICK_RED_BOX"))
    assert event.status == SequenceStatus.REPEATED_STEP


def test_experiment_completion(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    for action in ["PICK_RED_BOX", "PLACE_RED_BOX", "PICK_BLUE_BOX", "PLACE_BLUE_BOX"]:
        engine.submit_action(predict(action))
    assert not engine.is_finished()
    final_event = engine.submit_action(predict("COMPLETE_EXPERIMENT"))
    assert final_event.status == SequenceStatus.COMPLETE
    assert engine.is_finished()
    # Nothing further is processed once finished.
    assert engine.submit_action(predict("PICK_RED_BOX")) is None


def test_low_confidence_does_not_advance(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment, confidence_threshold=0.60)
    event = engine.submit_action(predict("PICK_RED_BOX", confidence=0.40))
    assert event.status == SequenceStatus.LOW_CONFIDENCE
    assert engine.current_step() == 1  # unchanged


def test_none_action_produces_no_event(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    assert engine.submit_action(ActionPrediction(action=None, confidence=0.0)) is None


def test_reset_clears_progress_and_history(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    engine.submit_action(predict("PICK_RED_BOX"))
    engine.submit_action(predict("PLACE_RED_BOX"))
    assert engine.current_step() == 3
    assert len(engine.history) == 2

    engine.reset()
    assert engine.current_step() == 1
    assert engine.history == []
    assert not engine.is_finished()
