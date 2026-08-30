"""Deterministic payload-drop evaluation used by the Gazebo mission.

Gazebo's x500 model in this workspace does not contain a releasable water
payload or a collision sensor for a splash.  This module provides the missing
evaluation layer: it propagates a virtual payload from the release point to a
configured horizontal target plane and reports whether its predicted impact is
inside the target radius.
"""

from dataclasses import asdict, dataclass
import math
from typing import Optional, Sequence


@dataclass(frozen=True)
class SimulatedDropResult:
    """One virtual release event expressed in the PX4 local-NED frame."""

    status: str
    flight_time_s: Optional[float]
    release_north_m: Optional[float]
    release_east_m: Optional[float]
    release_down_m: Optional[float]
    impact_north_m: Optional[float]
    impact_east_m: Optional[float]
    target_north_m: Optional[float]
    target_east_m: Optional[float]
    radial_error_m: Optional[float]
    hit_radius_m: float
    hit: bool
    message: str

    def as_dict(self):
        """Return a CSV/JSON-friendly representation."""
        return asdict(self)


class KinematicDropEvaluator:
    """Propagate a payload with constant horizontal velocity and gravity.

    The payload inherits the vehicle's local-NED velocity at release.  Optional
    wind values are represented as a constant horizontal drift velocity.  This
    deliberately models the quantity the controller can validate in the
    current simulator; it is not a substitute for fluid simulation.
    """

    def __init__(
        self,
        hit_radius_m: float,
        gravity_mps2: float = 9.80665,
        wind_north_mps: float = 0.0,
        wind_east_mps: float = 0.0,
    ):
        if hit_radius_m <= 0.0:
            raise ValueError("hit_radius_m must be positive")
        if gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        if not all(math.isfinite(value) for value in (
            hit_radius_m, gravity_mps2, wind_north_mps, wind_east_mps
        )):
            raise ValueError("drop-evaluation parameters must be finite")
        self.hit_radius_m = float(hit_radius_m)
        self.gravity_mps2 = float(gravity_mps2)
        self.wind_north_mps = float(wind_north_mps)
        self.wind_east_mps = float(wind_east_mps)

    def evaluate(
        self,
        release_ned: Sequence[float],
        velocity_ned: Sequence[float],
        target_ned: Sequence[float],
        impact_plane_down_m: float,
    ) -> SimulatedDropResult:
        """Predict an impact on ``down=impact_plane_down_m``.

        ``release_ned`` and ``target_ned`` may contain a third down component;
        only target north/east are used for the hit test.  The impact plane can
        be set from the latest depth target, avoiding an implicit assumption
        that the PX4 local-frame origin is exactly on the terrain surface.
        """
        if len(release_ned) != 3 or len(velocity_ned) != 3 or len(target_ned) != 3:
            raise ValueError("release_ned, velocity_ned and target_ned must each have three elements")
        values = tuple(release_ned) + tuple(velocity_ned) + tuple(target_ned) + (impact_plane_down_m,)
        if not all(math.isfinite(value) for value in values):
            return self._invalid("NONFINITE_INPUT")

        north, east, down = map(float, release_ned)
        vel_north, vel_east, vel_down = map(float, velocity_ned)
        target_north, target_east, _ = map(float, target_ned)
        height_m = float(impact_plane_down_m) - down
        if height_m <= 0.0:
            return self._invalid("INVALID_IMPACT_PLANE")

        # height = v_down * t + 0.5 * g * t^2 in a down-positive NED frame.
        discriminant = vel_down * vel_down + 2.0 * self.gravity_mps2 * height_m
        if discriminant < 0.0:
            return self._invalid("NO_BALLISTIC_SOLUTION")
        flight_time_s = (-vel_down + math.sqrt(discriminant)) / self.gravity_mps2
        if not math.isfinite(flight_time_s) or flight_time_s < 0.0:
            return self._invalid("INVALID_FLIGHT_TIME")

        impact_north = north + (vel_north + self.wind_north_mps) * flight_time_s
        impact_east = east + (vel_east + self.wind_east_mps) * flight_time_s
        radial_error_m = math.hypot(impact_north - target_north, impact_east - target_east)
        hit = radial_error_m <= self.hit_radius_m
        return SimulatedDropResult(
            status="HIT" if hit else "MISS",
            flight_time_s=flight_time_s,
            release_north_m=north,
            release_east_m=east,
            release_down_m=down,
            impact_north_m=impact_north,
            impact_east_m=impact_east,
            target_north_m=target_north,
            target_east_m=target_east,
            radial_error_m=radial_error_m,
            hit_radius_m=self.hit_radius_m,
            hit=hit,
            message="Predicted kinematic impact is inside the configured target radius." if hit else
                    "Predicted kinematic impact is outside the configured target radius.",
        )

    def unavailable(self, status: str, message: str) -> SimulatedDropResult:
        """Return a structured result when a release cannot be evaluated."""
        return SimulatedDropResult(
            status=status,
            flight_time_s=None,
            release_north_m=None,
            release_east_m=None,
            release_down_m=None,
            impact_north_m=None,
            impact_east_m=None,
            target_north_m=None,
            target_east_m=None,
            radial_error_m=None,
            hit_radius_m=self.hit_radius_m,
            hit=False,
            message=message,
        )

    def _invalid(self, status: str) -> SimulatedDropResult:
        return self.unavailable(
            status,
            "Insufficient valid flight or target data; no precision-hit conclusion was made.",
        )
