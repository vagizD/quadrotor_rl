"""Forward-Euler integration."""

from collections.abc import Callable
from dataclasses import dataclass

from common.types import FloatVector

from ._common import evaluate_derivative, require_state_vector, require_timestep


@dataclass(frozen=True)
class EulerIntegrator:
    """Fixed-step forward-Euler integrator."""

    dt: float

    def __post_init__(self) -> None:
        require_timestep(self.dt)

    def step(
        self,
        state: FloatVector,
        derivative_function: Callable[[FloatVector], FloatVector],
    ) -> FloatVector:
        require_state_vector(state, "state")
        derivative = evaluate_derivative(state, derivative_function)
        return state + self.dt * derivative
