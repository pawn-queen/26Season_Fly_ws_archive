import math

import numpy as np
import pytest

from detect.target_selection import (
    TARGET_OBSERVATION_FRAME_ID,
    align_depth_to_color,
    choose_highest_confidence_candidate,
    depth_scale_for_encoding,
    image_timestamps_within_skew,
    source_timestamp_matches_node_clock,
)


def candidate(confidence, depth_m, *, in_roi, distance_sq):
    return {
        'confidence': confidence,
        'depth_m': depth_m,
        'in_roi': in_roi,
        'center_distance_sq': distance_sq,
    }


@pytest.mark.parametrize('encoding', ['32FC1', '64FC1', ' 32fc1 '])
def test_float_depth_encodings_are_already_metres(encoding):
    assert depth_scale_for_encoding(encoding) == 1.0


@pytest.mark.parametrize('encoding', ['16UC1', 'MONO16', 'mono16'])
def test_integer_depth_encodings_are_converted_from_millimetres(encoding):
    assert depth_scale_for_encoding(encoding) == 0.001


def test_depth_dtype_is_used_when_encoding_is_empty():
    assert depth_scale_for_encoding('', 'f') == 1.0
    assert depth_scale_for_encoding('', 'u') == 0.001
    assert depth_scale_for_encoding('', 'i') == 0.001


def test_unknown_depth_representation_is_rejected():
    with pytest.raises(ValueError, match='Unsupported depth encoding'):
        depth_scale_for_encoding('', None)


def test_explicit_unsupported_encoding_does_not_silently_use_dtype():
    with pytest.raises(ValueError, match='Unsupported depth encoding'):
        depth_scale_for_encoding('8UC1', 'u')


def test_highest_confidence_target_can_be_reacquired_outside_roi():
    inside = candidate(0.80, 3.0, in_roi=True, distance_sq=4.0)
    outside = candidate(0.92, 3.2, in_roi=False, distance_sq=100.0)

    selected = choose_highest_confidence_candidate([inside, outside])

    assert selected is outside


def test_roi_only_mode_ignores_outside_target_even_when_confidence_is_higher():
    inside = candidate(0.80, 3.0, in_roi=True, distance_sq=4.0)
    outside = candidate(0.92, 3.2, in_roi=False, distance_sq=100.0)

    selected = choose_highest_confidence_candidate(
        [inside, outside],
        allow_outside_roi=False,
    )

    assert selected is inside


def test_invalid_depth_and_non_finite_confidence_are_not_selected():
    invalid_depth = candidate(0.99, 0.0, in_roi=True, distance_sq=1.0)
    nan_confidence = candidate(math.nan, 3.0, in_roi=True, distance_sq=1.0)
    usable = candidate(0.75, 3.1, in_roi=False, distance_sq=9.0)

    selected = choose_highest_confidence_candidate(
        [invalid_depth, nan_confidence, usable]
    )

    assert selected is usable


def test_confidence_ties_prefer_roi_then_nearest_image_centre():
    outside_near = candidate(0.90, 3.0, in_roi=False, distance_sq=1.0)
    inside_far = candidate(0.90, 3.0, in_roi=True, distance_sq=25.0)
    inside_near = candidate(0.90, 3.0, in_roi=True, distance_sq=4.0)

    selected = choose_highest_confidence_candidate(
        [outside_near, inside_far, inside_near]
    )

    assert selected is inside_near


def test_no_depth_valid_candidate_returns_none():
    candidates = [
        candidate(0.95, math.nan, in_roi=True, distance_sq=1.0),
        candidate(0.90, -1.0, in_roi=False, distance_sq=4.0),
    ]

    assert choose_highest_confidence_candidate(candidates) is None


def test_source_timestamp_is_used_when_clock_domains_match():
    assert source_timestamp_matches_node_clock(100.0, 100.2)
    assert source_timestamp_matches_node_clock(100.0, 110.0)


def test_receive_timestamp_is_needed_for_mismatched_or_zero_source_clock():
    assert not source_timestamp_matches_node_clock(100.0, 1_700_000_000.0)
    assert not source_timestamp_matches_node_clock(0.0, 100.0)


def test_rgb_depth_pair_requires_bounded_capture_time_skew():
    assert image_timestamps_within_skew(10.0, 10.03, 0.04)
    assert not image_timestamps_within_skew(10.0, 10.05, 0.04)
    assert not image_timestamps_within_skew(math.nan, 10.0, 0.04)


def test_identity_depth_registration_preserves_pixel_and_depth():
    depth = np.zeros((3, 3), dtype=float)
    depth[1, 1] = 2.0
    intrinsics = (2.0, 2.0, 1.0, 1.0)

    aligned = align_depth_to_color(
        depth,
        intrinsics,
        intrinsics,
        depth.shape,
    )

    assert aligned[1, 1] == pytest.approx(2.0)
    assert np.count_nonzero(aligned) == 1


def test_depth_registration_applies_baseline_in_color_optical_frame():
    depth = np.zeros((3, 3), dtype=float)
    depth[1, 1] = 2.0
    intrinsics = (2.0, 2.0, 1.0, 1.0)

    aligned = align_depth_to_color(
        depth,
        intrinsics,
        intrinsics,
        depth.shape,
        depth_to_color_translation=(1.0, 0.0, 0.0),
    )

    assert aligned[1, 2] == pytest.approx(2.0)
    assert aligned[1, 1] == 0.0


def test_depth_registration_uses_nearest_sample_as_z_buffer():
    depth = np.array([[1.0, 2.0]], dtype=float)

    aligned = align_depth_to_color(
        depth,
        (100.0, 1.0, 0.5, 0.0),
        (0.01, 1.0, 0.0, 0.0),
        (1, 1),
    )

    assert aligned[0, 0] == pytest.approx(1.0)


def test_empty_depth_registration_returns_empty_color_depth():
    aligned = align_depth_to_color(
        np.zeros((2, 2), dtype=float),
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (3, 4),
    )

    assert aligned.shape == (3, 4)
    assert not np.any(aligned)
    assert TARGET_OBSERVATION_FRAME_ID == 'target_camera_optical_frame'
