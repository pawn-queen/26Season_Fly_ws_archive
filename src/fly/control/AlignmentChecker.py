"""Continuous elapsed-time checker for horizontal target alignment."""

import math
import time


class AlignmentChecker:
    """Report alignment after the error remains below a threshold."""

    def __init__(
        self,
        logger_func,
        threshold=0.15,
        time_window=2.0,
        check_frequency=10,
        time_func=None,
    ):
        """
        Initialize the alignment threshold and continuous time window.

        ``check_frequency`` remains for API compatibility but is no longer used
        to convert time into a number of callback samples.  ``time_func`` may
        inject a monotonic or ROS clock for deterministic behaviour.
        """
        threshold = float(threshold)
        time_window = float(time_window)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be positive and finite")
        if not math.isfinite(time_window) or time_window < 0.0:
            raise ValueError("time_window must be non-negative and finite")

        self.logger_func = logger_func
        self.threshold = threshold
        self.time_window = time_window
        self.check_frequency = check_frequency
        self._time_func = time_func or time.monotonic
        self._within_threshold_since = None
        self._alignment_reported = False

    def check(self, current_x, current_y, target_x, target_y):
        """
        Check whether error stays below the threshold continuously.

        Return ``True`` once the continuous interval reaches ``time_window``.
        """
        error = math.hypot(current_x - target_x, current_y - target_y)
        now = self._time_func()

        if (
            not math.isfinite(error)
            or not math.isfinite(now)
            or error >= self.threshold
        ):
            self._within_threshold_since = None
            self._alignment_reported = False
            return False

        if (
            self._within_threshold_since is None
            or now < self._within_threshold_since
        ):
            self._within_threshold_since = now

        elapsed = now - self._within_threshold_since
        if elapsed < self.time_window:
            return False

        if not self._alignment_reported:
            self.logger_func(
                f"对准条件满足: 误差连续 {elapsed:.2f} 秒小于阈值 "
                f"{self.threshold} m。"
            )
            self._alignment_reported = True
        return True

    def reset(self):
        """Clear the continuous-alignment timer and reported state."""
        self._within_threshold_since = None
        self._alignment_reported = False
        self.logger_func("对准检查器已重置。")
