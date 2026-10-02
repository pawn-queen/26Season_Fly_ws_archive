"""Confidence-aware world-frame target anchoring for horizontal alignment."""

from collections import deque
import math


class TargetAnchorTracker:
    """
    Keep a stable NED anchor once it is locked.

    Rules
    -----
    * Initial lock: first observation whose confidence is at least
      ``initial_lock_confidence`` (default 0.8).
    * Same-bucket observations (within ``max_anchor_jump_m``) enter a
      rolling confidence window; the highest-confidence candidate in the
      window becomes the anchor.
    * Far observations (beyond ``max_anchor_jump_m``) are rejected
      unconditionally, UNLESS their confidence exceeds the current anchor's
      confidence by more than ``significant_confidence_margin``.  That
      exception lets a clearly better observation re-anchor the tracker.
    * Only ``reset()`` (observation stream break, timestamp reversal, task
      switch) can clear the anchor.
    """

    def __init__(
        self,
        confidence_window_s=4.0,
        hold_duration_s=2.5,
        max_anchor_jump_m=0.6,
        initial_lock_confidence=0.8,
        significant_confidence_margin=0.3,
    ):
        confidence_window_s = float(confidence_window_s)
        hold_duration_s = float(hold_duration_s)
        max_anchor_jump_m = float(max_anchor_jump_m)
        initial_lock_confidence = float(initial_lock_confidence)
        significant_confidence_margin = float(
            significant_confidence_margin
        )

        if not math.isfinite(confidence_window_s) or confidence_window_s <= 0.0:
            raise ValueError("confidence_window_s must be positive")
        if not math.isfinite(hold_duration_s) or hold_duration_s < 0.0:
            raise ValueError("hold_duration_s must be non-negative")
        if not math.isfinite(max_anchor_jump_m) or max_anchor_jump_m <= 0.0:
            raise ValueError("max_anchor_jump_m must be positive")
        if (
            not math.isfinite(initial_lock_confidence)
            or not 0.0 <= initial_lock_confidence <= 1.0
        ):
            raise ValueError(
                "initial_lock_confidence must be within [0, 1]"
            )
        if (
            not math.isfinite(significant_confidence_margin)
            or significant_confidence_margin < 0.0
        ):
            raise ValueError(
                "significant_confidence_margin must be non-negative"
            )

        self.confidence_window_s = confidence_window_s
        self.hold_duration_s = hold_duration_s
        self.max_anchor_jump_m = max_anchor_jump_m
        self.initial_lock_confidence = initial_lock_confidence
        self.significant_confidence_margin = significant_confidence_margin

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
        """Pick the highest-confidence candidate in the window."""
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
        """
        observed_at_s = float(observed_at_s)
        if not math.isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        target_ned = self._validated_ned(target_ned)
        self.last_observation_at_s = observed_at_s

        # Legacy Point-only path: replace the anchor directly.
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

        # --- Initial lock ---
        if self.anchor_ned is None:
            if confidence < self.initial_lock_confidence:
                return False
            self._candidates.clear()
            self._candidates.append((observed_at_s, confidence, target_ned))
            return self._select_best_candidate()

        # --- Anchor exists ---
        jump_m = math.hypot(
            target_ned[0] - self.anchor_ned[0],
            target_ned[1] - self.anchor_ned[1],
        )
        old_confidence = (
            self.anchor_confidence
            if self.anchor_confidence is not None
            else 0.0
        )

        if jump_m > self.max_anchor_jump_m:
            # Far observation.  Default: permanent reject.
            # Exception: a clearly better observation may re-anchor.
            if confidence > old_confidence + self.significant_confidence_margin:
                self._candidates.clear()
                self._candidates.append(
                    (observed_at_s, confidence, target_ned)
                )
                return self._select_best_candidate()
            return False

        # --- Same bucket: rolling confidence window ---
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