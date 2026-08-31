"""Pure helpers for selecting a single depth-assisted detection."""

import math

import numpy as np


TARGET_OBSERVATION_FRAME_ID = "target_camera_optical_frame"


def depth_scale_for_encoding(encoding, dtype_kind=None):
    """
    Return the multiplier that converts a depth image value to metres.

    ROS ``32FC1`` depth is already expressed in metres, while ``16UC1``
    depth is conventionally expressed in millimetres.  ``dtype_kind`` is a
    fallback for bridges which leave the encoding empty.
    """
    normalized = (encoding or "").strip().upper()
    if normalized in {"32FC1", "64FC1"}:
        return 1.0
    if normalized in {"16UC1", "MONO16"}:
        return 0.001
    if normalized:
        raise ValueError(
            "Unsupported depth encoding "
            f"{encoding!r} with dtype kind {dtype_kind!r}"
        )

    if dtype_kind == "f":
        return 1.0
    if dtype_kind in {"u", "i"}:
        return 0.001

    raise ValueError(
        "Unsupported depth encoding "
        f"{encoding!r} with dtype kind {dtype_kind!r}"
    )


def source_timestamp_matches_node_clock(
    source_stamp_s,
    received_at_s,
    max_clock_skew_s=3600.0,
):
    """
    Whether a sensor stamp can be matched to poses on the node clock.

    Gazebo images can carry simulation time while nodes still use system time.
    In that case the detector must stamp the observation with the time at which
    it received the frame, captured before inference, instead of forwarding an
    incomparable source timestamp.  The generous default deliberately keeps a
    delayed source stamp unchanged, allowing the controller to reject stale
    imagery rather than relabeling it as a new frame.
    """
    values = (source_stamp_s, received_at_s, max_clock_skew_s)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if source_stamp_s <= 0.0 or max_clock_skew_s < 0.0:
        return False
    clock_skew_s = abs(float(source_stamp_s) - float(received_at_s))
    return clock_skew_s <= max_clock_skew_s


def image_timestamps_within_skew(
    color_stamp_s,
    depth_stamp_s,
    max_skew_s,
):
    """Return whether RGB and depth capture timestamps form a valid pair."""
    try:
        color_stamp_s = float(color_stamp_s)
        depth_stamp_s = float(depth_stamp_s)
        max_skew_s = float(max_skew_s)
    except (TypeError, ValueError):
        return False
    if not all(
        math.isfinite(value)
        for value in (color_stamp_s, depth_stamp_s, max_skew_s)
    ):
        return False
    return (
        max_skew_s >= 0.0
        and abs(color_stamp_s - depth_stamp_s) <= max_skew_s
    )


def align_depth_to_color(
    depth_image_m,
    depth_intrinsics,
    color_intrinsics,
    output_shape,
    depth_to_color_rotation=None,
    depth_to_color_translation=None,
):
    """
    Project a metric depth image into the color optical image.

    Points and extrinsics use the ROS optical convention: x right, y down,
    z forward.  ``depth_to_color_translation`` is the depth-camera origin
    expressed in the color optical frame.  When several depth samples land on
    one color pixel, the nearest positive color-frame z value wins.
    """
    depth_image_m = np.asarray(depth_image_m, dtype=float)
    if depth_image_m.ndim != 2:
        raise ValueError("depth_image_m must be a two-dimensional image")

    try:
        output_height, output_width = (int(value) for value in output_shape)
    except (TypeError, ValueError):
        raise ValueError(
            "output_shape must contain height and width"
        ) from None
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_shape dimensions must be positive")

    def validated_intrinsics(values, name):
        try:
            fx, fy, cx, cy = (float(value) for value in values)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must contain fx, fy, cx, and cy"
            ) from None
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
            raise ValueError(f"{name} must contain finite values")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"{name} focal lengths must be positive")
        return fx, fy, cx, cy

    depth_fx, depth_fy, depth_cx, depth_cy = validated_intrinsics(
        depth_intrinsics,
        "depth_intrinsics",
    )
    color_fx, color_fy, color_cx, color_cy = validated_intrinsics(
        color_intrinsics,
        "color_intrinsics",
    )

    rotation = (
        np.eye(3, dtype=float)
        if depth_to_color_rotation is None
        else np.asarray(depth_to_color_rotation, dtype=float)
    )
    translation = (
        np.zeros(3, dtype=float)
        if depth_to_color_translation is None
        else np.asarray(depth_to_color_translation, dtype=float)
    )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("depth_to_color_rotation must be a finite 3x3 matrix")
    if (
        translation.shape != (3,)
        or not np.all(np.isfinite(translation))
    ):
        raise ValueError(
            "depth_to_color_translation must be a finite 3-vector"
        )

    aligned_depth = np.zeros((output_height, output_width), dtype=float)
    valid_mask = np.isfinite(depth_image_m) & (depth_image_m > 0.0)
    if not np.any(valid_mask):
        return aligned_depth

    depth_v, depth_u = np.nonzero(valid_mask)
    depth_z = depth_image_m[depth_v, depth_u]
    points_depth = np.column_stack((
        (depth_u - depth_cx) * depth_z / depth_fx,
        (depth_v - depth_cy) * depth_z / depth_fy,
        depth_z,
    ))
    points_color = points_depth @ rotation.T + translation
    color_z = points_color[:, 2]
    projectable = (
        np.all(np.isfinite(points_color), axis=1)
        & (color_z > 0.0)
    )
    if not np.any(projectable):
        return aligned_depth

    points_color = points_color[projectable]
    color_z = color_z[projectable]
    color_u = np.rint(
        color_fx * points_color[:, 0] / color_z + color_cx
    ).astype(np.int64)
    color_v = np.rint(
        color_fy * points_color[:, 1] / color_z + color_cy
    ).astype(np.int64)
    inside_image = (
        (color_u >= 0)
        & (color_u < output_width)
        & (color_v >= 0)
        & (color_v < output_height)
    )
    if not np.any(inside_image):
        return aligned_depth

    color_u = color_u[inside_image]
    color_v = color_v[inside_image]
    color_z = color_z[inside_image]
    flat_indices = color_v * output_width + color_u
    z_buffer = np.full(output_height * output_width, np.inf, dtype=float)
    np.minimum.at(z_buffer, flat_indices, color_z)
    populated = np.isfinite(z_buffer)
    aligned_depth.flat[populated] = z_buffer[populated]
    return aligned_depth


def choose_highest_confidence_candidate(candidates, allow_outside_roi=True):
    """
    Choose one finite, depth-valid candidate from the current frame.

    Confidence is the primary ordering key.  Exact confidence ties prefer a
    target inside the central ROI and then the target nearest the image centre.
    This keeps ROI behaviour as a tie-breaker without preventing reacquisition
    after wind or tracking error moves the target outside the ROI.
    """
    eligible = []
    for candidate in candidates:
        confidence = candidate.get("confidence")
        depth_m = candidate.get("depth_m")
        if not (
            math.isfinite(confidence)
            and math.isfinite(depth_m)
            and depth_m > 0.0
        ):
            continue
        if not allow_outside_roi and not candidate.get("in_roi", False):
            continue
        eligible.append(candidate)

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda candidate: (
            candidate["confidence"],
            bool(candidate.get("in_roi", False)),
            -float(candidate.get("center_distance_sq", math.inf)),
        ),
    )
