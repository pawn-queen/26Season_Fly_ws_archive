"""Safe OpenCV GUI capability checks for local and remote sessions."""

import os
import subprocess
import sys


_GUI_PROBE_CODE = """
import cv2
import numpy as np

frame = np.zeros((1, 1, 3), dtype=np.uint8)
cv2.imshow('__control_gui_probe__', frame)
cv2.waitKey(1)
cv2.destroyAllWindows()
"""


def opencv_gui_available(timeout_s=5.0):
    """
    Probe OpenCV GUI support in a child process.

    Qt may abort the whole process when a stale WSL/X11 display is present, so
    checking ``DISPLAY`` alone or catching ``cv2.error`` in the flight process
    is insufficient.  The child process contains that failure.
    """
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    if os.name != 'nt' and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    ):
        return False

    try:
        result = subprocess.run(
            [sys.executable, '-c', _GUI_PROBE_CODE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
