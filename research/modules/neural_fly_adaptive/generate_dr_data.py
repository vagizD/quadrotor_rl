"""Generate Domain Randomization dataset."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from robots.quadrotor.params import QuadrotorParams
from research.simulator.integrators.rk4 import RK4Integrator
from software.pipeline.control.px4.px4 import PX4Controller
from research.simulator.environment.disturbances import WindForce, AerodynamicDrag
from research.projects.px4_baseline.run_scenarios import run_scenario

def main():
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "px4_hover.toml"
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
    out_dir = Path(__file__).resolve().parent / "runs" / "px4_scenarios" / "dr_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 10 different wind conditions
    scales = np.linspace(0.0, 2.0, 10)
    
    controller = PX4Controller(params)
    
    for i, scale in enumerate(scales):
        base_wind = np.array([1.5, 0.5, 0.0]) * scale
        gust = 2.0 * scale
        noise = 0.2 * scale
        
        disturbances = [
            WindForce(constant_force=base_wind, gust_amplitude=gust, noise_std=noise),
            AerodynamicDrag(linear_drag_coeff=0.2, angular_drag_coeff=0.05)
        ]
        
        # We simulate collecting data with the baseline PX4 controller
        # We use a random scenario mix to get diverse states
        scenario = ["hover", "point_to_point", "circle"][i % 3]
        run_scenario(
            f"{scenario}_dr_{i}", params, integrator, config, out_dir, 
            controller=controller, disturbances=disturbances,
            disturbance_scale=scale, wind_vector=base_wind / params.mass
        )
        
    print(f"Generated {len(scales)} DR trajectories in {out_dir}")

if __name__ == "__main__":
    main()
