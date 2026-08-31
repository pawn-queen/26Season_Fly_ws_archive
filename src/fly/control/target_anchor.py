"""Confidence-aware world-frame target anchoring for horizontal alignment."""

from collections import deque
import math


class TargetAnchorTracker:
    """
    Keep the highest-confidence recent target as a fixed NED anchor.

    The controller converts each *new* camera observation to NED before adding
    it here.  Reusing the resulting world point prevents an old camera-relative
    vector from moving with the aircraft while it is being blown or commanded
    horizontally.
    """

    def __init__(self, confidence_window_s=4.0, hold_duration_s=2.5):
        """Initialize the selection window and short target-loss hold time."""
        confidence_window_s = float(confidence_window_s)
        hold_duration_s = float(hold_duration_s)
        if (
            not math.isfinite(confidence_window_s)
            or confidence_window_s <= 0.0
        ):
            raise ValueError("confidence_window_s must be positive")
        if not math.isfinite(hold_duration_s) or hold_duration_s < 0.0:
            raise ValueError("hold_duration_s must be non-negative")

        self.confidence_window_s = confidence_window_s
        self.hold_duration_s = hold_duration_s
        self._candidates = deque()
        self.anchor_ned = None
        self.anchor_confidence = None
        self.anchor_observed_at_s = None
        self.last_observation_at_s = None

    @staticmethod
    def _validated_ned(target_ned):
        values = tuple(float(value) for value in target_ned)
        if (
            len(values) != 3
            or not all(math.isfinite(value) for value in values)
        ):
            raise ValueError(
                "target_ned must contain three finite coordinates"
            )
        return values

    def _prune(self, now_s):
        cutoff_s = now_s - self.confidence_window_s
        while self._candidates and self._candidates[0][0] < cutoff_s:
            self._candidates.popleft()

    def _select_best_candidate(self):
        if not self._candidates:
            return False

        observed_at_s, confidence, target_ned = max(
            self._candidates,
            key=lambda candidate: (candidate[1], candidate[0]),
        )
        old_anchor = self.anchor_ned
        self.anchor_ned = target_ned
        self.anchor_confidence = confidence
        self.anchor_observed_at_s = observed_at_s
        return old_anchor != self.anchor_ned

    def add_observation(self, target_ned, observed_at_s, confidence=None):
        """
        Add one NED observation and return whether the selected anchor changed.

        A finite confidence participates in the rolling highest-confidence
        selection.  ``None`` preserves compatibility with the legacy Point-only
        topic by making that observation the current anchor directly.
        """
        observed_at_s = float(observed_at_s)
        if not math.isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        target_ned = self._validated_ned(target_ned)
        self.last_observation_at_s = observed_at_s

        if confidence is None:
            old_anchor = self.anchor_ned
            self._candidates.clear()
            self.anchor_ned = target_ned
            self.anchor_confidence = None
            self.anchor_observed_at_s = observed_at_s
            return old_anchor != self.anchor_ned

        confidence = float(confidence)
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite or None")

        self._candidates.append((observed_at_s, confidence, target_ned))
        self._prune(observed_at_s)
        return self._select_best_candidate()

    def refresh(self, now_s):
        """Expire old candidates; return whether selected anchor changed."""
        now_s = float(now_s)
        self._prune(now_s)
        return self._select_best_candidate()

    def is_active(self, now_s):
        """Whether commands may still advance toward the cached anchor."""
        if self.anchor_ned is None or self.last_observation_at_s is None:
            return False
        age_s = self.latest_observation_age_s(now_s)
        return math.isfinite(age_s) and age_s <= self.hold_duration_s

    def latest_observation_age_s(self, now_s):
        """Return observation age, or infinity for absent/future data."""
        if self.last_observation_at_s is None:
            return math.inf
        age_s = float(now_s) - self.last_observation_at_s
        if not math.isfinite(age_s) or age_s < 0.0:
            return math.inf
        return age_s

    def reset(self):
        """Clear all candidates and the selected anchor."""
        self._candidates.clear()
        self.anchor_ned = None
        self.anchor_confidence = None
        self.anchor_observed_at_s = None
        self.last_observation_at_s = None


def px4_pose_attitude_timestamps_match(
    position_timestamp_us,
    attitude_timestamp_us,
    max_skew_s,
):
    """Return whether two positive PX4 sample timestamps are synchronized."""
    try:
        position_timestamp_us = float(position_timestamp_us)
        attitude_timestamp_us = float(attitude_timestamp_us)
        max_skew_s = float(max_skew_s)
    except (TypeError, ValueError):
        return False

    values = (position_timestamp_us, attitude_timestamp_us, max_skew_s)
    if not all(math.isfinite(value) for value in values):
        return False
    if position_timestamp_us <= 0.0 or attitude_timestamp_us <= 0.0:
        return False
    if max_skew_s <= 0.0:
        return False
    return (
        abs(position_timestamp_us - attitude_timestamp_us)
        <= max_skew_s * 1e6
    )


def altitude_within_threshold(current_z, target_z, threshold):
    """Return whether a finite NED altitude is within a positive threshold."""
    try:
        current_z = float(current_z)
        target_z = float(target_z)
        threshold = float(threshold)
    except (TypeError, ValueError):
        return False

    if not all(
        math.isfinite(value)
        for value in (current_z, target_z, threshold)
    ):
        return False
    return threshold > 0.0 and abs(current_z - target_z) < threshold
