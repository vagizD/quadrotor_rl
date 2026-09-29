"""Shared validation for numerical integrators."""

from collections.abc import Callable

import numpy as np

from geometry.common.types import FloatVector


def require_state_vector(value: FloatVector, name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def require_timestep(dt: float) -> None:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")


def evaluate_derivative(
    state: FloatVector,
    derivative_function: Callable[[FloatVector], FloatVector],
) -> FloatVector:
    derivative = derivative_function(state)
    require_state_vector(derivative, "derivative")
    if derivative.shape != state.shape:
        raise ValueError(
            "derivative must have the same shape as state, "
            f"got {derivative.shape} and {state.shape}"
        )
    return derivative
