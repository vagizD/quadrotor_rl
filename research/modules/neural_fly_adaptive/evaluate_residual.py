"""Evaluate ResidualPX4Controller across varying disturbance levels."""

import sys
from pathlib import Path
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from geometry.math.rotation import Quaternion
from robots.quadrotor.params import QuadrotorParams
from geometry.state import RigidBodyState, RigidBodyStateDerivative
from robots.quadrotor.dynamics import compute_state_derivative
from software.pipeline.control.px4.px4 import PX4Controller
from software.pipeline.control.residual.residual_px4 import ResidualPX4Controller
from research.simulator.integrators.rk4 import RK4Integrator
from research.simulator.environment.disturbances import WindForce, AerodynamicDrag

def run_eval_scenario(params, integrator, config, controller, disturbances):
    horizon = config["environment"]["episode_horizon"]
    steps = int(horizon / integrator.dt)
    
    state = RigidBodyState(
        position=np.array([1.0, 1.0, 0.0]),
        velocity=np.zeros(3),
        quaternion=Quaternion.identity(),
        angular_velocity=np.zeros(3)
    )
    
    target_pos = np.array([0.0, 0.0, 2.0])
    target_yaw = config["environment"]["target_heading"]
    
    t = 0.0
    for i in range(steps):
        action = controller.compute_control(state, target_pos, target_yaw, dt=integrator.dt)
        thrusts = action * params.max_thrust
        
        def deriv_fn(s_vec):
            s = RigidBodyState.from_vector(s_vec)
            total_force = np.zeros(3)
            total_torque = np.zeros(3)
            for dist in disturbances:
                f, tau = dist.apply(s, thrusts, params, t)
                total_force += f
                total_torque += tau
                
            return compute_state_derivative(
                s, thrusts, params, 
                external_force=total_force, 
                external_torque=total_torque
            ).to_vector()

        next_state_vec = integrator.step(state.to_vector(), deriv_fn)
        state = RigidBodyState.from_vector(next_state_vec)
        state.quaternion = state.quaternion.normalized()
        t += integrator.dt

    return np.linalg.norm(target_pos - state.position)

def main():
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "research" / "projects" / "px4_baseline" / "px4_hover.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    q_cfg = config["quadrotor"]
    params = QuadrotorParams(
        mass=q_cfg["mass"],
        arm_length=q_cfg["arm_length"],
        inertia=np.array(q_cfg["inertia"]),
        yaw_torque_coefficient=q_cfg["yaw_torque_coefficient"],
        max_thrust=q_cfg["max_thrust"],
        gravity=q_cfg["gravity"],
        motor_spin_directions=np.array(q_cfg["motor_spin_directions"])
    )

    integrator = RK4Integrator(dt=config["simulation"]["dt"])
    
    px4 = PX4Controller(params)
    model_path = Path(__file__).resolve().parent.parent.parent.parent / "research" / "projects" / "models" / "residual_model.pt"
    
    # Disturbance levels: Scale from 0.0 (no disturbance) to 3.0 (extreme)
    scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    
    print("| Disturbance Scale | Base Wind X | Gust Amp | Noise Std | PX4 Error (m) | ResidualPX4 Error (m) | Improvement (m) |")
    print("|---|---|---|---|---|---|---|")
    
    for scale in scales:
        base_wind = np.array([1.5, 0.5, 0.0]) * scale
        gust = 2.0 * scale
        noise = 0.2 * scale
        
        res_px4 = ResidualPX4Controller(params, model_path)
        
        # We keep drag constant because drag is always present, we only scale the external weather conditions
        disturbances = [
            WindForce(constant_force=base_wind, gust_amplitude=gust, noise_std=noise),
            AerodynamicDrag(linear_drag_coeff=0.2, angular_drag_coeff=0.05)
        ]
        
        px4_err = run_eval_scenario(params, integrator, config, px4, disturbances)
        res_err = run_eval_scenario(params, integrator, config, res_px4, disturbances)
        
        diff = px4_err - res_err
        print(f"| {scale:.2f}x | {base_wind[0]:.2f}N | {gust:.2f}N | {noise:.2f} | {px4_err:.4f} | {res_err:.4f} | {diff:.4f} |")

if __name__ == "__main__":
    main()
