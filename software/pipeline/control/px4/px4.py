"""PX4-style cascaded controller for Quadrotor."""

import numpy as np

from geometry.common.types import FloatVector
from geometry.math.rotation import Quaternion
from robots.quadrotor.params import QuadrotorParams
from geometry.state.state import RigidBodyState, RigidBodyStateDerivative


class PIDController:
    """A simple 3D PID controller."""

    def __init__(self, kp: list[float], ki: list[float], kd: list[float]):
        self.kp = np.array(kp, dtype=np.float64)
        self.ki = np.array(ki, dtype=np.float64)
        self.kd = np.array(kd, dtype=np.float64)
        self.integral = np.zeros(3, dtype=np.float64)
        self.prev_error = np.zeros(3, dtype=np.float64)

    def step(self, error: FloatVector, dt: float) -> FloatVector:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0.0 else np.zeros_like(error)
        self.prev_error = error.copy()
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def rot_to_quat(R: np.ndarray) -> Quaternion:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return Quaternion(w, x, y, z)

class PX4Controller:
    """Deterministic PX4-style cascaded controller."""

    def __init__(self, params: QuadrotorParams, config: dict | None = None):
        self.params = params
        
        # Default tuning (can be overridden by config later)
        self.pos_p = np.array([1.2, 1.2, 1.2], dtype=np.float64)
        self.vel_pid = PIDController(
            kp=[2.0, 2.0, 2.0],
            ki=[0.0, 0.0, 0.5],
            kd=[0.2, 0.2, 0.2]
        )
        
        self.att_p = np.array([6.0, 6.0, 3.0], dtype=np.float64)
        self.rate_pid = PIDController(
            kp=[0.15, 0.15, 0.1],
            ki=[0.0, 0.0, 0.0],
            kd=[0.005, 0.005, 0.0]
        )
        
        # Build Mixer matrix
        arm = self.params.arm_length
        yaw_coef = self.params.yaw_torque_coefficient
        dirs = self.params.motor_spin_directions
        
        # Mapping from [f1, f2, f3, f4] to [T, tau_x, tau_y, tau_z]
        M = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -arm, 0.0, arm],
            [-arm, 0.0, arm, 0.0],
            [yaw_coef * dirs[0], yaw_coef * dirs[1], yaw_coef * dirs[2], yaw_coef * dirs[3]]
        ], dtype=np.float64)
        self.M_inv = np.linalg.inv(M)

    def compute_control(
        self, 
        state: RigidBodyState, 
        target_position: FloatVector, 
        target_yaw: float, 
        dt: float,
        delta_F: FloatVector | None = None,
        delta_tau: FloatVector | None = None
    ) -> FloatVector:
        """Compute [0, 1] normalized motor commands."""
        
        # 1. Position -> Velocity Setpoint
        pos_error = target_position - state.position
        vel_sp = self.pos_p * pos_error
        
        # 2. Velocity -> Acceleration Setpoint
        vel_error = vel_sp - state.velocity
        acc_sp = self.vel_pid.step(vel_error, dt)
        
        # Compensate for external force disturbance in world frame
        if delta_F is not None:
            acc_sp -= delta_F / self.params.mass
            
        # Gravity compensation
        acc_sp[2] += self.params.gravity
        
        # 3. Acceleration -> Thrust and Attitude Setpoint
        thrust_sp = self.params.mass * np.linalg.norm(acc_sp)
        
        acc_norm = np.linalg.norm(acc_sp)
        if acc_norm > 1e-6:
            z_b_des = acc_sp / acc_norm
        else:
            z_b_des = np.array([0.0, 0.0, 1.0])
            
        x_c = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        y_norm = np.linalg.norm(y_b_des)
        if y_norm > 1e-6:
            y_b_des /= y_norm
        else:
            y_b_des = np.array([0.0, 1.0, 0.0])
            
        x_b_des = np.cross(y_b_des, z_b_des)
        
        R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
        q_des = rot_to_quat(R_des)
        
        # 4. Attitude -> Rate Setpoint
        q_curr = state.quaternion
        q_inv = Quaternion(q_curr.w, -q_curr.x, -q_curr.y, -q_curr.z)
        q_err = q_inv * q_des
        
        # Shortest path
        sign = 1.0 if q_err.w >= 0 else -1.0
        rate_sp = 2.0 * sign * self.att_p * np.array([q_err.x, q_err.y, q_err.z])
        
        # 5. Rate -> Torque Setpoint
        rate_err = rate_sp - state.angular_velocity
        torque_sp = self.rate_pid.step(rate_err, dt)
        
        # Compensate for external torque disturbance in body frame
        if delta_tau is not None:
            torque_sp -= delta_tau
            
        # 6. Control Allocation (Mixer)
        wrench = np.array([thrust_sp, torque_sp[0], torque_sp[1], torque_sp[2]])
        thrusts = self.M_inv @ wrench
        
        # Convert to [0, 1] and clip
        normalized_cmds = thrusts / self.params.max_thrust
        return np.clip(normalized_cmds, 0.0, 1.0)
