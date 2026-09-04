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
    # The recognizer (Phase 4) never emits COMPLETE_EXPERIMENT — there's no
    # physical gesture for it — so only the 4 real PICK/PLACE actions are
    # submitted; the engine auto-advances through the trailing COMPLETE step.
    sequence = ["PICK_RED_BOX", "PLACE_RED_BOX", "PICK_BLUE_BOX", "PLACE_BLUE_BOX"]
    statuses = []
    for action in sequence:
        event = engine.submit_action(predict(action))
        assert event is not None
        statuses.append(event.status)
    assert statuses[:-1] == [SequenceStatus.CORRECT] * 3
    assert statuses[-1] == SequenceStatus.COMPLETE
    assert engine.is_finished()


def test_wrong_object(sample_experiment):
    # Same action TYPE (PICK) as expected, different object, that object's
    # own step is still ahead — grabbed the wrong item for this step.
    engine = RuleBasedSequenceEngine(sample_experiment)
    event = engine.submit_action(predict("PICK_BLUE_BOX"))  # step 1 expects PICK_RED_BOX
    assert event.status == SequenceStatus.WRONG_OBJECT
    assert event.expected == "PICK_RED_BOX"
    assert event.detected == "PICK_BLUE_BOX"
    assert engine.current_step() == 1  # did not advance


def test_skipped_step(sample_experiment):
    # 1 -> 3 (step 2, PLACE_RED_BOX, never happened): a different action
    # TYPE (PICK) than what's expected (PLACE), matching a real but later
    # step — the current step was skipped over, not substituted.
    engine = RuleBasedSequenceEngine(sample_experiment)
    engine.submit_action(predict("PICK_RED_BOX"))  # step 1, correct
    event = engine.submit_action(predict("PICK_BLUE_BOX"))  # skips step 2 (PLACE_RED_BOX)
    assert event.status == SequenceStatus.SKIPPED_STEP
    assert engine.current_step() == 2


def test_out_of_sequence_unknown_action(sample_experiment):
    # True fallback: detected_key doesn't correspond to any known step at
    # all. Shouldn't normally happen (the recognizer only emits known
    # action_keys) — kept as a defensive case.
    engine = RuleBasedSequenceEngine(sample_experiment)
    event = engine.submit_action(predict("JUMP_ROPE"))
    assert event.status == SequenceStatus.OUT_OF_SEQUENCE
    assert engine.current_step() == 1


def test_recovery(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    engine.submit_action(predict("PICK_RED_BOX"))
    error_event = engine.submit_action(predict("PICK_BLUE_BOX"))
    assert error_event.status == SequenceStatus.SKIPPED_STEP

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
    for action in ["PICK_RED_BOX", "PLACE_RED_BOX", "PICK_BLUE_BOX"]:
        engine.submit_action(predict(action))
    assert not engine.is_finished()
    # PLACE_BLUE_BOX is the last real gesture — the engine auto-advances
    # through COMPLETE_EXPERIMENT itself and returns *that* event.
    final_event = engine.submit_action(predict("PLACE_BLUE_BOX"))
    assert final_event.status == SequenceStatus.COMPLETE
    assert engine.is_finished()
    # Nothing further is processed once finished.
    assert engine.submit_action(predict("PICK_RED_BOX")) is None


def test_complete_auto_advance_logs_both_events(sample_experiment):
    engine = RuleBasedSequenceEngine(sample_experiment)
    for action in ["PICK_RED_BOX", "PLACE_RED_BOX", "PICK_BLUE_BOX"]:
        engine.submit_action(predict(action))
    returned_event = engine.submit_action(predict("PLACE_BLUE_BOX"))

    # submit_action can only return one event, but both the real step's
    # CORRECT outcome and the auto-generated COMPLETE event must still be
    # in history for logging/reporting (Phase 8).
    assert returned_event.status == SequenceStatus.COMPLETE
    assert len(engine.history) == 5
    assert engine.history[-2].status == SequenceStatus.CORRECT
    assert engine.history[-2].detected == "PLACE_BLUE_BOX"
    assert engine.history[-1].status == SequenceStatus.COMPLETE
    assert engine.history[-1].detected == "COMPLETE_EXPERIMENT"


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
