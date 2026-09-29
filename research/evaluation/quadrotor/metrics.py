"""Evaluation metrics for quadrotor trajectories."""

import numpy as np
from research.pipeline.planning.trajectory import Trajectory

def compute_position_error(trajectory: Trajectory) -> float:
    """Compute the final position error between the quadrotor and the target."""
    final_pos = trajectory.states[-1, :3]
    target_pos = trajectory.target_positions[-1]
    return float(np.linalg.norm(final_pos - target_pos))

def compute_rmse(trajectory: Trajectory) -> float:
    """Compute the Root Mean Square Error (RMSE) over the trajectory."""
    positions = trajectory.states[:, :3]
    targets = trajectory.target_positions
    errors = np.linalg.norm(positions - targets, axis=1)
    return float(np.sqrt(np.mean(errors**2)))

def print_evaluation_report(scenario_name: str, trajectory: Trajectory) -> dict[str, float | np.ndarray]:
    """Print and return common metrics for a trajectory."""
    final_pos = trajectory.states[-1, :3]
    target_pos = trajectory.target_positions[-1]
    pos_error = compute_position_error(trajectory)
    rmse = compute_rmse(trajectory)
    
    print(f"--- Scenario: {scenario_name} ---")
    print(f"Final position:  {final_pos}")
    print(f"Final target:    {target_pos}")
    print(f"Position Error:  {pos_error:.4f} m")
    print(f"RMSE:            {rmse:.4f} m\n")
    
    return {
        "final_position": final_pos,
        "final_target": target_pos,
        "position_error": pos_error,
        "rmse": rmse
    }
