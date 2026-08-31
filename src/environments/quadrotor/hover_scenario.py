"""Deterministic initial states for hover-policy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.types import FloatVector
from geometry.rotation import Quaternion
from robots.quadrotor import QuadrotorState


@dataclass(frozen=True)
class HoverScenario:
    """One named hover initial-state configuration."""

    name: str
    position_offset: tuple[float, float, float]
    description: str

    def initial_state(
        self,
        target_position: FloatVector,
        target_heading: float = 0.0,
    ) -> QuadrotorState:
        """Create a state with only the configured position offset."""
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(
                "target_position must have shape (3,), "
                f"got {target.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain only finite values")
        if not np.isfinite(target_heading):
            raise ValueError("target_heading must be finite")

        offset = np.asarray(self.position_offset, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("position_offset must be a finite 3-vector")

        half_heading = float(target_heading) / 2.0
        heading = Quaternion(
            np.cos(half_heading),
            0.0,
            0.0,
            np.sin(half_heading),
        )
        return QuadrotorState(
            position=target + offset,
            velocity=np.zeros(3, dtype=np.float64),
            quaternion=heading,
            angular_velocity=np.zeros(3, dtype=np.float64),
        )


HOVER_SCENARIOS = (
    HoverScenario(
        name="at_target",
        position_offset=(0.0, 0.0, 0.0),
        description="Start exactly at the target with the target attitude.",
    ),
    HoverScenario(
        name="near_target",
        position_offset=(0.25, 0.0, 0.0),
        description="Start 0.25 m along world +X from the target.",
    ),
    HoverScenario(
        name="far_target",
        position_offset=(2.0, 0.0, 0.0),
        description="Start 2.0 m along world +X from the target.",
    ),
)


def get_hover_scenarios() -> tuple[HoverScenario, ...]:
    """Return all predefined hover diagnostic scenarios."""
    return HOVER_SCENARIOS
