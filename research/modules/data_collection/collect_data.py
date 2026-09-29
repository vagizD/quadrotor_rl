import os
import argparse
from pathlib import Path
import numpy as np
import tomli as tomllib

from robots.quadrotor.params import QuadrotorParams
from robots.quadrotor.model import Quadrotor
from geometry.state import RigidBodyState
from geometry.math.rotation import Quaternion
from software.pipeline.control.px4.px4 import PX4Controller
from research.simulator.simulator import QuadrotorEnvironment
from research.simulator.integrators.rk4 import RK4Integrator
from research.pipeline.planning.trajectory import Trajectory
from research.simulator.environment.disturbances import WindForce

from research.evaluation.quadrotor.scenarios.hover import HoverScenario
from research.evaluation.quadrotor.scenarios.point_to_point import PointToPointScenario
from research.evaluation.quadrotor.scenarios.circle import CircleScenario

def main():
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
    
    integrator = RK4Integrator(dt=0.01)
    
    out_dir = Path("research/modules/data/domain_randomization_400")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_episodes = 400
    for i in range(num_episodes):
        # 1. No wind, 2. Small wind, 3. Medium wind, 4. Strong wind
        wind_type = i % 4
        
        if wind_type == 0:
            wind = np.zeros(3)
            wind_label = "no_wind"
        elif wind_type == 1:
            wind = np.random.uniform(-2.0, 2.0, size=3)
            wind_label = "small_wind"
        elif wind_type == 2:
            wind = np.random.uniform(-5.0, 5.0, size=3)
            wind_label = "medium_wind"
        else:
            wind = np.random.uniform(-10.0, 10.0, size=3)
            wind_label = "strong_wind"
            
        disturbance = WindForce(constant_force=wind.tolist())
        
        initial_state = RigidBodyState(
            position=np.random.uniform(-2, 2, size=3),
            velocity=np.random.uniform(-1, 1, size=3),
            quaternion=Quaternion.identity(),
            angular_velocity=np.random.uniform(-0.5, 0.5, size=3)
        )
        
        env = QuadrotorEnvironment(
            params=params,
            integrator=integrator,
            initial_state=initial_state,
            disturbances=[disturbance]
        )
        
        scenarios = [
            HoverScenario(env),
            PointToPointScenario(env),
            CircleScenario(env)
        ]
        
        scenario = scenarios[i % len(scenarios)]
        controller = PX4Controller(params)
        
        print(f"Collecting episode {i+1}/{num_episodes} - {scenario.name} with {wind_label} {wind}")
        
        # We also need to log disturbance forces!
        states = []
        target_positions = []
        actions = []
        times = []
        dist_f = []
        dist_t = []
        
        target_fn = scenario.get_target_trajectory()
        num_steps = int(scenario.duration / scenario.dt)
        
        for _ in range(num_steps):
            t = env.time
            current_target = target_fn(t)

            states.append(env.quadrotor.state.to_vector())
            target_positions.append(current_target)
            times.append(t)

            action = controller.compute_control(env.quadrotor.state, current_target, 0.0, scenario.dt)
            thrusts = action * env.params.max_thrust
            actions.append(thrusts)
            
            # calculate disturbance ground truth
            ext_f = np.zeros(3)
            ext_t = np.zeros(3)
            for d in env.disturbances:
                f, tau = d.apply(env.quadrotor.state, thrusts, env.params, t)
                ext_f += f
                ext_t += tau
                
            dist_f.append(ext_f)
            dist_t.append(ext_t)

            env.step(thrusts)
            
        t = env.time
        states.append(env.quadrotor.state.to_vector())
        target_positions.append(target_fn(t))
        times.append(t)

        trajectory = Trajectory(
            states=np.array(states),
            target_positions=np.array(target_positions),
            thrusts=np.array(actions),
            times=np.array(times),
            disturbance_forces=np.array(dist_f),
            disturbance_torques=np.array(dist_t),
            wind_vector=wind,
            wind_label=wind_label # Need to add this to Trajectory!
        )
        
        npz_path = out_dir / f"episode_{i:03d}.npz"
        trajectory.save(npz_path)

if __name__ == "__main__":
    main()
