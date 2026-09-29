"""Classical fourth-order Runge-Kutta integration."""

from collections.abc import Callable
from dataclasses import dataclass

from geometry.common.types import FloatVector

from ._common import evaluate_derivative, require_state_vector, require_timestep


@dataclass(frozen=True)
class RK4Integrator:
    """Fixed-step classical fourth-order Runge-Kutta integrator."""

    dt: float

    def __post_init__(self) -> None:
        require_timestep(self.dt)

    def step(
        self,
        state: FloatVector,
        derivative_function: Callable[[FloatVector], FloatVector],
    ) -> FloatVector:
        require_state_vector(state, "state")
        k1 = evaluate_derivative(state, derivative_function)

        state_2 = state + 0.5 * self.dt * k1
        require_state_vector(state_2, "intermediate state")
        k2 = evaluate_derivative(state_2, derivative_function)

        state_3 = state + 0.5 * self.dt * k2
        require_state_vector(state_3, "intermediate state")
        k3 = evaluate_derivative(state_3, derivative_function)

        state_4 = state + self.dt * k3
        require_state_vector(state_4, "intermediate state")
        k4 = evaluate_derivative(state_4, derivative_function)

        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
