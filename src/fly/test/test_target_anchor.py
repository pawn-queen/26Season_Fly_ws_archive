import pytest

from control.target_anchor import (
    TargetAnchorTracker,
    altitude_within_threshold,
    px4_pose_attitude_timestamps_match,
)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_highest_confidence_nearby_observation_wins(selection_mode):
    tracker = TargetAnchorTracker(
        confidence_window_s=3.0, selection_mode=selection_mode
    )

    assert tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.80)
    assert not tracker.add_observation((1.2, 0.0, 2.0), 1.0, 0.70)
    assert tracker.anchor_ned == (1.0, 0.0, 2.0)

    assert tracker.add_observation((1.4, 0.0, 2.0), 2.0, 0.95)
    assert tracker.anchor_ned == (1.4, 0.0, 2.0)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_newer_observation_breaks_equal_confidence_tie(selection_mode):
    tracker = TargetAnchorTracker(
        confidence_window_s=3.0, selection_mode=selection_mode
    )
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.80)
    tracker.add_observation((1.2, 0.0, 2.0), 1.0, 0.80)

    assert tracker.anchor_ned == (1.2, 0.0, 2.0)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_expired_best_candidate_yields_to_recent_candidate(selection_mode):
    tracker = TargetAnchorTracker(
        confidence_window_s=2.0, selection_mode=selection_mode
    )
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.95)
    tracker.add_observation((1.2, 0.0, 2.0), 1.0, 0.80)

    assert not tracker.refresh(2.0)
    assert tracker.refresh(2.1)
    assert tracker.anchor_ned == (1.2, 0.0, 2.0)
    assert not tracker.refresh(3.1)
    assert tracker.anchor_ned == (1.2, 0.0, 2.0)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_anchor_hold_uses_latest_observation_even_if_best_is_older(selection_mode):
    tracker = TargetAnchorTracker(
        confidence_window_s=3.0,
        hold_duration_s=2.0,
        selection_mode=selection_mode,
    )
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.95)
    tracker.add_observation((1.2, 0.0, 2.0), 1.0, 0.80)

    assert tracker.is_active(2.9)
    assert tracker.is_active(3.0)
    assert not tracker.is_active(3.1)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_legacy_observation_becomes_anchor_immediately(selection_mode):
    tracker = TargetAnchorTracker(
        confidence_window_s=3.0, selection_mode=selection_mode
    )
    tracker.add_observation((1.0, 0.0, 2.0), 0.0, 0.95)

    assert tracker.add_observation((4.0, 5.0, 2.0), 1.0, None)
    assert tracker.anchor_ned == (4.0, 5.0, 2.0)
    assert tracker.anchor_confidence is None
    assert not tracker.refresh(2.0)
    assert tracker.anchor_ned == (4.0, 5.0, 2.0)


def test_future_observation_is_not_treated_as_active_after_clock_rewind():
    tracker = TargetAnchorTracker(confidence_window_s=4.0, hold_duration_s=2.5)
    tracker.add_observation((1.0, 0.0, 2.0), 10.0, 0.95)

    assert tracker.latest_observation_age_s(9.0) == float('inf')
    assert not tracker.is_active(9.0)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_new_detection_reacquires_after_controller_resets_expired_anchor(selection_mode):
    tracker = TargetAnchorTracker(
        confidence_window_s=4.0,
        hold_duration_s=2.5,
        selection_mode=selection_mode,
    )
    tracker.add_observation((1.0, 2.0, 3.0), 0.0, 0.90)

    assert not tracker.is_active(2.6)
    assert not tracker.add_observation((4.0, 5.0, 3.0), 3.0, 0.95)
    assert tracker.anchor_ned == (1.0, 2.0, 3.0)
    tracker.reset()
    assert tracker.anchor_ned is None
    assert tracker.anchor_confidence is None
    assert tracker.anchor_observed_at_s is None
    assert tracker.latest_observation_age_s(3.0) == float('inf')
    assert tracker.add_observation((4.0, 5.0, 3.0), 3.0, 0.95)
    assert tracker.is_active(3.0)
    assert tracker.anchor_ned == (4.0, 5.0, 3.0)


def test_default_window_and_hold_match_controller_defaults():
    tracker = TargetAnchorTracker()

    assert tracker.confidence_window_s == 4.0
    assert tracker.hold_duration_s == 2.5
    assert tracker.selection_mode == "max-confidence"
    assert tracker.max_anchor_jump_m == 0.6
    assert tracker.initial_lock_confidence == 0.8
    assert tracker.significant_confidence_margin == 0.3


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_initial_lock_requires_confidence_threshold(selection_mode):
    tracker = TargetAnchorTracker(selection_mode=selection_mode)

    assert not tracker.add_observation((0.0, 0.0, 2.0), 0.0, 0.799999)
    assert tracker.anchor_ned is None
    assert not tracker.is_active(0.0)
    assert tracker.add_observation((0.0, 0.0, 2.0), 1.0, 0.8)
    assert tracker.is_active(1.0)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_jump_gate_includes_boundary_and_rejects_far_high_confidence(selection_mode):
    tracker = TargetAnchorTracker(selection_mode=selection_mode)
    tracker.add_observation((0.0, 0.0, 2.0), 0.0, 0.8)

    assert not tracker.add_observation((0.600001, 0.0, 2.0), 1.0, 1.0)
    assert tracker.anchor_ned == (0.0, 0.0, 2.0)
    assert tracker.add_observation((0.6, 0.0, 2.0), 2.0, 0.9)
    assert tracker.anchor_ned == (0.6, 0.0, 2.0)


@pytest.mark.parametrize("selection_mode", ("max-confidence", "top25"))
def test_far_reanchor_requires_strict_confidence_margin_and_clears_history(selection_mode):
    tracker = TargetAnchorTracker(
        confidence_window_s=2.0,
        initial_lock_confidence=0.5,
        selection_mode=selection_mode,
    )
    for index in range(5):
        tracker.add_observation((0.0, 0.0, 2.0), index * 0.1, 0.5)

    assert not tracker.add_observation((5.0, 0.0, 2.0), 1.0, 0.8)
    assert tracker.anchor_ned == (0.0, 0.0, 2.0)
    assert tracker.add_observation((5.0, 0.0, 2.0), 2.0, 0.800001)
    assert tracker.anchor_ned == (5.0, 0.0, 2.0)
    assert not tracker.add_observation((5.2, 0.0, 2.0), 3.0, 0.7)
    assert tracker.refresh(4.1)
    assert tracker.anchor_ned == (5.2, 0.0, 2.0)


@pytest.mark.parametrize(
    "sample_count, expected_ned, expected_confidence, expected_time",
    (
        (4, (0.02, 0.01, 2.0), 0.95, 1.0),
        (5, (0.04, 0.02, 4.5), 0.93, 3.0),
        (8, (0.06, 0.03, 5.5), 0.945, 5.0),
        (9, (0.10, 0.05, 6.0), 0.94, 7.0),
    ),
)
def test_top25_uses_ceiling_quartile_and_coordinate_medians(
    sample_count, expected_ned, expected_confidence, expected_time
):
    tracker = TargetAnchorTracker(
        confidence_window_s=100.0, selection_mode="top25"
    )
    confidences = (0.8, 0.95, 0.85, 0.91, 0.89, 0.94, 0.86, 0.93, 0.90)
    altitudes = (4.0, 2.0, 3.0, 7.0, 1.0, 9.0, 0.0, 6.0, 5.0)
    for index in range(sample_count):
        tracker.add_observation(
            (index * 0.02, index * 0.01, altitudes[index]),
            float(index),
            confidences[index],
        )

    assert tracker.anchor_ned == pytest.approx(expected_ned)
    assert tracker.anchor_confidence == pytest.approx(expected_confidence)
    assert tracker.anchor_observed_at_s == expected_time


def test_top25_prefers_newer_samples_when_quartile_boundary_has_confidence_tie():
    tracker = TargetAnchorTracker(selection_mode="top25")
    for index in range(5):
        tracker.add_observation((index * 0.1, 0.0, 2.0), float(index), 0.9)

    assert tracker.anchor_ned == pytest.approx((0.35, 0.0, 2.0))
    assert tracker.anchor_confidence == 0.9
    assert tracker.anchor_observed_at_s == 4.0


def test_tracker_rejects_unknown_selection_mode():
    with pytest.raises(ValueError, match="selection_mode"):
        TargetAnchorTracker(selection_mode="average")


def test_px4_pose_and_attitude_timestamps_must_be_synchronized():
    assert px4_pose_attitude_timestamps_match(1_000_000, 1_050_000, 0.10)
    assert px4_pose_attitude_timestamps_match(1_000_000, 1_100_000, 0.10)
    assert not px4_pose_attitude_timestamps_match(1_000_000, 1_100_001, 0.10)
    assert not px4_pose_attitude_timestamps_match(1_000_000, 1_200_000, 0.10)
    assert not px4_pose_attitude_timestamps_match(0, 1_000_000, 0.10)
    assert not px4_pose_attitude_timestamps_match(1_000_000, None, 0.10)
    assert not px4_pose_attitude_timestamps_match(1_000_000, 1_000_000, 0.0)
    assert not px4_pose_attitude_timestamps_match(float('inf'), 1_000_000, 0.10)
    assert not px4_pose_attitude_timestamps_match(1_000_000, 1_000_000, float('nan'))


def test_altitude_gate_rejects_invalid_and_out_of_range_values():
    assert altitude_within_threshold(-2.05, -2.0, 0.10)
    assert not altitude_within_threshold(-1.8, -2.0, 0.10)
    assert not altitude_within_threshold(float('nan'), -2.0, 0.10)
    assert not altitude_within_threshold(-2.0, None, 0.10)
    assert not altitude_within_threshold(-2.0, -2.0, 0.0)
    assert not altitude_within_threshold(-1.875, -2.0, 0.125)
    assert altitude_within_threshold(-1.875001, -2.0, 0.125)
    assert not altitude_within_threshold(-2.0, -2.0, float('inf'))


def test_tracker_rejects_non_finite_durations():
    with pytest.raises(ValueError):
        TargetAnchorTracker(confidence_window_s=float('nan'))
    with pytest.raises(ValueError):
        TargetAnchorTracker(hold_duration_s=float('inf'))
