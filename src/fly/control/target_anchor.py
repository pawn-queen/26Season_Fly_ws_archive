"""Confidence-aware world-frame target anchoring for horizontal alignment."""

from collections import deque
from dataclasses import dataclass
import math
from statistics import median


@dataclass(frozen=True)
class ObservationResult:
    """Outcome of ingesting a point, independent of whether its anchor won."""

    accepted: bool
    anchor_changed: bool
    lock_acquired: bool = False
    anchor_reselected: bool = False


class TargetAnchorTracker:
    """
    Select recent confidence-bearing targets as a fixed NED anchor.

    The controller converts each *new* camera observation to NED before adding
    it here.  Reusing the resulting world point prevents an old camera-relative
    vector from moving with the aircraft while it is being blown or commanded
    horizontally.  By default the highest-confidence observation wins; the
    optional top25 mode uses the coordinate-wise median of the top quartile.
    The newest mode selects the latest observation meeting its own threshold.
    The ned-fix mode freezes the selected point near the bucket and reselects
    only after fresh observations continuously disagree with that point.
    """

    def __init__(
        self,
        confidence_window_s=4.0,
        hold_duration_s=2.5,
        selection_mode="max-confidence",
        lock_enabled=False,
        newest_confidence_threshold=0.9,
        ned_fix_distance=0.08,
        ned_fix_time_s=1.0,
        ned_fix_release_distance=0.08,
        ned_fix_release_time_s=1.0,
        observation_timeout_s=1.2,
    ):
        """Initialize the selection mode, window, and target-loss hold time."""
        confidence_window_s = float(confidence_window_s)
        hold_duration_s = float(hold_duration_s)
        if (
            not math.isfinite(confidence_window_s)
            or confidence_window_s <= 0.0
        ):
            raise ValueError("confidence_window_s must be positive")
        if not math.isfinite(hold_duration_s) or hold_duration_s < 0.0:
            raise ValueError("hold_duration_s must be non-negative")
        if selection_mode not in ("max-confidence", "top25", "newest", "ned-fix"):
            raise ValueError(
                "selection_mode must be 'max-confidence', 'top25', 'newest', or 'ned-fix'"
            )
        if selection_mode == "newest":
            newest_confidence_threshold = float(newest_confidence_threshold)
            if (
                not math.isfinite(newest_confidence_threshold)
                or not 0.0 <= newest_confidence_threshold <= 1.0
            ):
                raise ValueError("newest_confidence_threshold must be in [0, 1]")
        if selection_mode == "ned-fix":
            ned_fix_distance = float(ned_fix_distance)
            ned_fix_time_s = float(ned_fix_time_s)
            ned_fix_release_distance = float(ned_fix_release_distance)
            ned_fix_release_time_s = float(ned_fix_release_time_s)
            observation_timeout_s = float(observation_timeout_s)
            for name, value in (
                ("ned_fix_distance", ned_fix_distance),
                ("ned_fix_release_distance", ned_fix_release_distance),
                ("observation_timeout_s", observation_timeout_s),
            ):
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be positive and finite")
            for name, value in (
                ("ned_fix_time_s", ned_fix_time_s),
                ("ned_fix_release_time_s", ned_fix_release_time_s),
            ):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be non-negative and finite")

        self.confidence_window_s = confidence_window_s
        self.hold_duration_s = hold_duration_s
        self.selection_mode = selection_mode
        self.lock_enabled = lock_enabled
        self.newest_confidence_threshold = newest_confidence_threshold
        self.ned_fix_distance = ned_fix_distance
        self.ned_fix_time_s = ned_fix_time_s
        self.ned_fix_release_distance = ned_fix_release_distance
        self.ned_fix_release_time_s = ned_fix_release_time_s
        self.observation_timeout_s = observation_timeout_s
        self.ned_fixed = False
        self.latest_target_ned = None
        self._ned_fix_started_at_s = None
        self._ned_fix_release_started_at_s = None
        self._ned_fix_selection_since_s = None
        self._ned_fix_last_control_at_s = None
        self.lock_ned = None
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
        candidates = self._candidates
        if self.selection_mode == "ned-fix":
            if self.ned_fixed:
                return False
            if self._ned_fix_selection_since_s is not None:
                # Keep the original pool intact for the independent 1 m lock.
                candidates = [
                    candidate for candidate in candidates
                    if candidate[0] >= self._ned_fix_selection_since_s
                ]
        if not candidates:
            return False

        if self.selection_mode in ("max-confidence", "ned-fix"):
            observed_at_s, confidence, target_ned = max(
                candidates,
                key=lambda candidate: (candidate[1], candidate[0]),
            )
        elif self.selection_mode == "newest":
            observed_at_s, confidence, target_ned = max(
                self._candidates,
                key=lambda candidate: candidate[0],
            )
        else:
            ranked_candidates = sorted(
                self._candidates,
                key=lambda candidate: (candidate[1], candidate[0]),
                reverse=True,
            )
            selected_count = max(
                1,
                math.ceil(len(ranked_candidates) * 0.25),
            )
            selected_candidates = ranked_candidates[:selected_count]
            observed_at_s = max(
                candidate[0] for candidate in selected_candidates
            )
            confidence = median(
                candidate[1] for candidate in selected_candidates
            )
            target_ned = tuple(
                median(candidate[2][axis] for candidate in selected_candidates)
                for axis in range(3)
            )
        old_anchor = self.anchor_ned
        self.anchor_ned = target_ned
        self.anchor_confidence = confidence
        self.anchor_observed_at_s = observed_at_s
        return old_anchor != self.anchor_ned

    def _update_lock(self):
        if self.lock_enabled and self._candidates:
            self.lock_ned = max(
                self._candidates,
                key=lambda candidate: (candidate[1], candidate[0]),
            )[2]
        else:
            self.lock_ned = None

    def reset_ned_fix_timers(self):
        """Interrupt confirmation without releasing a briefly lost fixed point."""
        self._ned_fix_started_at_s = None
        self._ned_fix_release_started_at_s = None

    def _expire_ned_fix(self, now_s):
        # Ingress may commit a newer image after the control cycle read its
        # clock. That older cycle timestamp must not expire the fixed point.
        if self.last_observation_at_s is not None and now_s < self.last_observation_at_s:
            return
        age_s = self.latest_observation_age_s(now_s)
        if age_s >= self.observation_timeout_s:
            self.reset_ned_fix_timers()
        if age_s > self.hold_duration_s:
            self.ned_fixed = False
            self.reset_ned_fix_timers()

    def update_ned_fix(self, now_s, distance_m, eligible):
        """Freeze the current anchor after continuous valid control conditions."""
        if self.selection_mode != "ned-fix":
            return False
        now_s = float(now_s)
        if not math.isfinite(now_s) or (
            self._ned_fix_last_control_at_s is not None
            and now_s < self._ned_fix_last_control_at_s
        ):
            self.reset()
            return False
        self._ned_fix_last_control_at_s = now_s
        self._expire_ned_fix(now_s)
        if self.ned_fixed:
            return False
        if (
            not eligible or self.anchor_ned is None
            or not math.isfinite(distance_m)
            or distance_m >= self.ned_fix_distance
            or self.latest_observation_age_s(now_s) >= self.observation_timeout_s
            or not self.is_active(now_s)
        ):
            self._ned_fix_started_at_s = None
            return False
        if self._ned_fix_started_at_s is None:
            self._ned_fix_started_at_s = now_s
        if now_s - self._ned_fix_started_at_s < self.ned_fix_time_s:
            return False
        self.ned_fixed = True
        self.reset_ned_fix_timers()
        return True

    def _ingest_ned_fix(self, target_ned, observed_at_s, confidence, now_s):
        """Process a fresh point without allowing selection to move a fixed point."""
        if confidence is None:
            if self.lock_enabled:
                return ObservationResult(False, False)
        else:
            confidence = float(confidence)
            if not math.isfinite(confidence):
                raise ValueError("confidence must be finite or None")
        now_s = observed_at_s if now_s is None else float(now_s)
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if (
            not 0.0 <= now_s - observed_at_s < self.observation_timeout_s
            or (self.last_observation_at_s is not None
                and observed_at_s <= self.last_observation_at_s)
        ):
            return ObservationResult(False, False)
        self._prune(now_s)
        self._update_lock()
        lock_acquired = False
        if self.lock_enabled:
            if observed_at_s < now_s - self.confidence_window_s:
                return ObservationResult(False, False)
            if self.lock_ned is not None and math.hypot(
                target_ned[0] - self.lock_ned[0],
                target_ned[1] - self.lock_ned[1],
            ) > 1.0:
                return ObservationResult(False, False)
            lock_acquired = self.lock_ned is None

        old_anchor = self.anchor_ned
        if self.latest_observation_age_s(now_s) > self.hold_duration_s:
            self._ned_fix_selection_since_s = observed_at_s
        self._expire_ned_fix(now_s)
        self.latest_target_ned = target_ned
        self.last_observation_at_s = observed_at_s
        if confidence is None:
            self._candidates.clear()
            if not self.ned_fixed:
                self.anchor_ned = target_ned
                self.anchor_confidence = None
                self.anchor_observed_at_s = observed_at_s
        else:
            self._candidates.append((observed_at_s, confidence, target_ned))
            self._update_lock()
            self._select_best_candidate()

        reselected = False
        if self.ned_fixed:
            deviation_m = math.hypot(
                target_ned[0] - self.anchor_ned[0],
                target_ned[1] - self.anchor_ned[1],
            )
            if deviation_m > self.ned_fix_release_distance:
                if self._ned_fix_release_started_at_s is None:
                    self._ned_fix_release_started_at_s = observed_at_s
                if (
                    observed_at_s - self._ned_fix_release_started_at_s
                    >= self.ned_fix_release_time_s
                ):
                    self.ned_fixed = False
                    self.reset_ned_fix_timers()
                    self._ned_fix_selection_since_s = observed_at_s
                    self.anchor_ned = target_ned
                    self.anchor_confidence = confidence
                    self.anchor_observed_at_s = observed_at_s
                    reselected = True
            else:
                self._ned_fix_release_started_at_s = None
        return ObservationResult(
            True, old_anchor != self.anchor_ned, lock_acquired, reselected,
        )

    def add_observation(self, target_ned, observed_at_s, confidence=None):
        """
        Add one NED observation and return whether the selected anchor changed.

        A finite confidence participates in the configured rolling selection.
        When unlocked, ``None`` preserves compatibility with the legacy
        Point-only topic by making that observation the current anchor directly.
        Use ingest_observation() when acceptance must be distinguished from an
        unchanged anchor.
        """
        return self.ingest_observation(
            target_ned, observed_at_s, confidence
        ).anchor_changed

    def ingest_observation(
        self, target_ned, observed_at_s, confidence=None, now_s=None
    ):
        """Return acceptance separately from anchor changes and lock acquisition.

        Locked observations are gated against the pre-insertion max-confidence
        point in the same rolling pool. Expiry uses current time, not arrival's
        image timestamp. Rejected points never renew the accepted observation age.
        """
        observed_at_s = float(observed_at_s)
        if not math.isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        target_ned = self._validated_ned(target_ned)

        if self.selection_mode == "ned-fix":
            return self._ingest_ned_fix(
                target_ned, observed_at_s, confidence, now_s,
            )

        if confidence is None:
            if self.lock_enabled or self.selection_mode == "newest":
                return ObservationResult(False, False)
            self.last_observation_at_s = observed_at_s
            old_anchor = self.anchor_ned
            self._candidates.clear()
            self.anchor_ned = target_ned
            self.anchor_confidence = None
            self.anchor_observed_at_s = observed_at_s
            return ObservationResult(True, old_anchor != self.anchor_ned)

        confidence = float(confidence)
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite or None")
        if (
            self.selection_mode == "newest"
            and confidence < self.newest_confidence_threshold
        ):
            return ObservationResult(False, False)

        lock_acquired = False
        if self.lock_enabled:
            now_s = observed_at_s if now_s is None else float(now_s)
            if not math.isfinite(now_s):
                raise ValueError("now_s must be finite")
            self._prune(now_s)
            self._update_lock()
            if (
                observed_at_s < now_s - self.confidence_window_s
                or observed_at_s > now_s
            ):
                return ObservationResult(False, False)
            if self.lock_ned is not None and math.hypot(
                target_ned[0] - self.lock_ned[0],
                target_ned[1] - self.lock_ned[1],
            ) > 1.0:
                return ObservationResult(False, False)
            lock_acquired = self.lock_ned is None

        self.last_observation_at_s = observed_at_s
        self._candidates.append((observed_at_s, confidence, target_ned))
        self._prune(observed_at_s)
        self._update_lock()
        return ObservationResult(True, self._select_best_candidate(), lock_acquired)

    def refresh(self, now_s):
        """Expire old candidates; return whether selected anchor changed."""
        now_s = float(now_s)
        if self.selection_mode == "ned-fix":
            if not math.isfinite(now_s) or (
                self._ned_fix_last_control_at_s is not None
                and now_s < self._ned_fix_last_control_at_s
            ):
                had_anchor = self.anchor_ned is not None
                self.reset()
                return had_anchor
            self._ned_fix_last_control_at_s = now_s
            self._expire_ned_fix(now_s)
        self._prune(now_s)
        self._update_lock()
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
        self.ned_fixed = False
        self.latest_target_ned = None
        self.reset_ned_fix_timers()
        self._ned_fix_selection_since_s = None
        self._ned_fix_last_control_at_s = None
        self._candidates.clear()
        self.lock_ned = None
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
