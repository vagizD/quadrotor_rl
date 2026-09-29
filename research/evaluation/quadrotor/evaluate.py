import os
import argparse
from pathlib import Path
import numpy as np

from robots.quadrotor.params import QuadrotorParams
from robots.quadrotor.model import Quadrotor
from research.simulator.simulator import QuadrotorEnvironment
from software.pipeline.control.px4.px4 import PX4Controller
from research.evaluation.quadrotor.metrics import print_evaluation_report
from research.simulator.visualization.rerun_viewer import log_trajectory

from research.evaluation.quadrotor.scenarios.hover import HoverScenario
from research.evaluation.quadrotor.scenarios.hover_with_wind import HoverWithWindScenario
from research.evaluation.quadrotor.scenarios.point_to_point import PointToPointScenario
from research.evaluation.quadrotor.scenarios.circle import CircleScenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="research/evaluation/quadrotor/metrics")
    args = parser.parse_args()

    base_out_dir = Path(args.out_dir)
    
    import tomli as tomllib
    config_path = Path("configs/research/modules/px4_config/px4_hover.toml")
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
    
    from geometry.state import RigidBodyState
    from geometry.math.rotation import Quaternion
    
    from research.simulator.integrators.rk4 import RK4Integrator
    integrator = RK4Integrator(dt=0.01)
    
    def create_env(scenario_class):
        initial_state = RigidBodyState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=Quaternion.identity(),
            angular_velocity=np.zeros(3)
        )
        quad = Quadrotor(params=params, state=initial_state)
        env = QuadrotorEnvironment(
            params=params,
            integrator=integrator,
            initial_state=initial_state
        )
        return scenario_class(env)

    scenarios = [
        create_env(HoverScenario),
        create_env(HoverWithWindScenario),
        create_env(PointToPointScenario),
        create_env(CircleScenario),
    ]
    
    from software.pipeline.control.residual.residual_px4 import NeuralFlyPX4Controller, MLPResidualPX4Controller, RLResidualPX4Controller
    import torch
    
    nf_path = Path("research/modules/neural_fly_adaptive/models/residual_model.pt")
    mlp_path = Path("research/modules/mlp_residual/models/mlp_residual_model.pt")
    rl_path = Path("research/modules/rl_residual/models/rl_residual_model.zip")
    
    controllers = [
        ("PX4Baseline", PX4Controller(params)),
    ]
    
    if nf_path.exists():
        controllers.append(("NeuralFlyPX4", NeuralFlyPX4Controller(params, nf_path, num_basis=16, lr=0.01)))
    
    if mlp_path.exists():
        controllers.append(("MLPResidualPX4", MLPResidualPX4Controller(params, mlp_path, history_len=10)))
        
    if rl_path.exists():
        from stable_baselines3 import PPO
        rl_model = PPO.load(str(rl_path))
        controllers.append(("RLResidualPX4", RLResidualPX4Controller(params, rl_model, history_len=10)))
    
    for scenario in scenarios:
        for ctrl_name, controller in controllers:
            print(f"\n--- Running Scenario: {scenario.name} | Controller: {ctrl_name} ---")
            
            # Reset controller state if it's residual
            if hasattr(controller, "a_hat"):
                controller.a_hat = np.zeros(controller.num_basis)
            if hasattr(controller, "state_history"):
                controller.state_history.extend([np.zeros(13)] * controller.history_len)
                controller.action_history.extend([np.zeros(4)] * controller.history_len)
                controller.error_history.extend([np.zeros(3)] * controller.history_len)
                
            trajectory = scenario.run(controller)
            
            # Calculate and print metrics
            metrics = print_evaluation_report(f"{scenario.name} ({ctrl_name})", trajectory)
            
            # Save artifacts
            scenario_out_dir = base_out_dir / scenario.name
            scenario_out_dir.mkdir(parents=True, exist_ok=True)
            
            trajectory.save(scenario_out_dir / f"{scenario.name}_{ctrl_name}_trajectory.npz")
            
            log_trajectory(
                trajectory=trajectory,
                arm_length=params.arm_length,
                recording_path=scenario_out_dir / f"{scenario.name}_{ctrl_name}_trajectory.rrd",
                step_stride=1
            )
            print(f"Artifacts saved to {scenario_out_dir}")

if __name__ == "__main__":
    main()
