import pytest

from control.target_anchor import (
    TargetAnchorTracker,
    altitude_within_threshold,
    px4_pose_attitude_timestamps_match,
)


def test_highest_confidence_recent_observation_wins():
    tracker = TargetAnchorTracker(confidence_window_s=3.0)

    assert tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.80)
    assert not tracker.add_observation((2.0, 0.0, 2.0), 1.0, 0.70)
    assert tracker.anchor_ned == (1.0, 0.0, 2.0)

    assert tracker.add_observation((3.0, 0.0, 2.0), 2.0, 0.95)
    assert tracker.anchor_ned == (3.0, 0.0, 2.0)


def test_newer_observation_breaks_equal_confidence_tie():
    tracker = TargetAnchorTracker(confidence_window_s=3.0)
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.80)
    tracker.add_observation((2.0, 0.0, 2.0), 1.0, 0.80)

    assert tracker.anchor_ned == (2.0, 0.0, 2.0)


def test_expired_best_candidate_yields_to_recent_candidate():
    tracker = TargetAnchorTracker(confidence_window_s=2.0)
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.95)
    tracker.add_observation((2.0, 0.0, 2.0), 1.0, 0.80)

    assert tracker.refresh(2.1)
    assert tracker.anchor_ned == (2.0, 0.0, 2.0)


def test_anchor_hold_uses_latest_observation_even_if_best_is_older():
    tracker = TargetAnchorTracker(confidence_window_s=3.0, hold_duration_s=2.0)
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.95)
    tracker.add_observation((2.0, 0.0, 2.0), 1.0, 0.80)

    assert tracker.is_active(2.9)
    assert not tracker.is_active(3.1)


def test_legacy_observation_becomes_anchor_immediately():
    tracker = TargetAnchorTracker(confidence_window_s=3.0)
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.95)

    assert tracker.add_observation((4.0, 5.0, 2.0), 1.0, None)
    assert tracker.anchor_ned == (4.0, 5.0, 2.0)
    assert tracker.anchor_confidence is None


def test_future_observation_is_not_treated_as_active_after_clock_rewind():
    tracker = TargetAnchorTracker(confidence_window_s=4.0, hold_duration_s=2.5)
    tracker.add_observation((1.0, 0.0, 2.0), 10.0, 0.95)

    assert tracker.latest_observation_age_s(9.0) == float('inf')
    assert not tracker.is_active(9.0)


def test_new_detection_reacquires_after_cached_anchor_hold_expires():
    tracker = TargetAnchorTracker(confidence_window_s=4.0, hold_duration_s=2.5)
    tracker.add_observation((1.0, 2.0, 3.0), 0.0, 0.90)

    assert not tracker.is_active(2.6)
    assert tracker.add_observation((4.0, 5.0, 3.0), 3.0, 0.95)
    assert tracker.is_active(3.0)
    assert tracker.anchor_ned == (4.0, 5.0, 3.0)


def test_default_window_and_hold_match_controller_defaults():
    tracker = TargetAnchorTracker()

    assert tracker.confidence_window_s == 4.0
    assert tracker.hold_duration_s == 2.5


def test_px4_pose_and_attitude_timestamps_must_be_synchronized():
    assert px4_pose_attitude_timestamps_match(1_000_000, 1_050_000, 0.10)
    assert not px4_pose_attitude_timestamps_match(1_000_000, 1_200_000, 0.10)
    assert not px4_pose_attitude_timestamps_match(0, 1_000_000, 0.10)
    assert not px4_pose_attitude_timestamps_match(1_000_000, None, 0.10)


def test_altitude_gate_rejects_invalid_and_out_of_range_values():
    assert altitude_within_threshold(-2.05, -2.0, 0.10)
    assert not altitude_within_threshold(-1.8, -2.0, 0.10)
    assert not altitude_within_threshold(float('nan'), -2.0, 0.10)
    assert not altitude_within_threshold(-2.0, None, 0.10)
    assert not altitude_within_threshold(-2.0, -2.0, 0.0)


def test_tracker_rejects_non_finite_durations():
    with pytest.raises(ValueError):
        TargetAnchorTracker(confidence_window_s=float('nan'))
    with pytest.raises(ValueError):
        TargetAnchorTracker(hold_duration_s=float('inf'))
