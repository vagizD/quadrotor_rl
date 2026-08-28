# Description — Pure-RL Quadrotor Hovering

## 1. Purpose

The goal is to study and demonstrate stable quadrotor hovering using a **pure reinforcement-learning controller** inside a custom 3D rigid-body simulator.

The project has two conceptual components:

1. a deterministic quadrotor environment that evolves according to rigid-body dynamics; and
2. a learned feedback policy that maps the current state and target directly to four rotor thrust commands.

The conventional PX4 cascaded controller is used only as a reference architecture for understanding the control problem. It is not part of the final control loop.

The final learned control law is

```text
current state + desired hover setpoint
                    |
                    v
                RL policy
                    |
                    v
             f1, f2, f3, f4
```

where each `f_i` is the thrust produced by one rotor.

---

## 2. Closed-Loop Pipeline

The complete system is

```text
desired hover setpoint
          |
          v
current state ----------------------+
          |                         |
          v                         |
      RL policy                     |
          |                         |
          v                         |
   motor thrusts f1..f4             |
          |                         |
          v                         |
 quadrotor rigid-body dynamics      |
          |                         |
          v                         |
     state derivative               |
          |                         |
          v                         |
  numerical time integration        |
          |                         |
          v                         |
       next state ------------------+
```

The responsibilities are conceptually distinct:

```text
policy:
(current state, desired state) -> motor thrusts

physics:
(current state, motor thrusts) -> state derivative

integration:
(current state, state derivative, dt) -> next state
```

The policy does not directly predict the next state or the state derivative.

---

## 3. State

The quadrotor state is

```text
x = (p_W, v_W, q_WB, omega_B)
```

where

```text
p_W      position in the world frame
v_W      linear velocity in the world frame
q_WB     unit quaternion describing body orientation relative to world
omega_B  angular velocity expressed in the body frame
```

The state contains 13 scalar values:

```text
3 position
3 velocity
4 quaternion
3 angular velocity
```

Linear acceleration and angular acceleration are not stored as state. They are computed from the current state and motor thrusts.

---

## 4. Action

The policy action is the four rotor thrust magnitudes

```text
u = (f1, f2, f3, f4)
```

with

```text
f_i >= 0
```

The thrust values are the actuator-level control quantities for the initial model.

The first version does not model electrical motor dynamics or rotor-speed transients. A commanded rotor thrust therefore acts directly in the rigid-body dynamics.

The policy is intentionally direct:

```text
(state, target) -> four rotor thrusts
```

There is no PID, attitude controller, angular-rate controller, control-allocation block, or other classical controller between the policy and the rotors.

---

## 5. Reference Frames

Two right-handed frames are used:

```text
W: fixed world frame
B: body-fixed quadrotor frame
```

The project uses

```text
+z_W upward
+z_B along the total rotor thrust direction
```

Gravity is therefore

```text
g_W = [0, 0, -g]^T
```

The rotation matrix

```text
R_WB(q)
```

maps a vector from body coordinates to world coordinates:

```text
v_W = R_WB v_B
```

The body thrust axis in world coordinates is

```text
z_B_in_W = R_WB [0, 0, 1]^T
```

Exact sign, motor-numbering, and quaternion conventions are fixed separately in `conventions.md`.

---

## 6. Translational Dynamics

The total rotor thrust in body coordinates is

```text
F_B = [0, 0, T]^T
```

where

```text
T = f1 + f2 + f3 + f4
```

This body-frame force must be rotated into the world frame.

The translational equations are

```text
p_dot_W = v_W
```

and

```text
m v_dot_W =
    m g_W
    + R_WB(q) F_B
    + F_ext,W
```

or equivalently

```text
v_dot_W =
    g_W
    + (1/m) R_WB(q) F_B
    + (1/m) F_ext,W
```

where

```text
m         quadrotor mass
F_ext,W   optional external force expressed in the world frame
```

The external-force term can represent disturbances such as wind.

For nominal hovering with no external disturbance and an upright vehicle,

```text
v_W = 0
q = identity orientation
T = m g
```

gives zero translational acceleration.

---

## 7. Rotational Dynamics

The quadrotor inertia matrix is

```text
J
```

expressed in the body frame.

The rotational rigid-body equation is

```text
J omega_dot
    = tau
    - omega x (J omega)
```

and therefore

```text
omega_dot
    = J^-1 [tau - omega x (J omega)]
```

where

```text
omega   body-frame angular velocity
tau     total body-frame torque
```

The cross term

```text
omega x (J omega)
```

appears because angular momentum is being described in the rotating body frame.

For the initial symmetric quadrotor model,

```text
J = diag(J_x, J_y, J_z)
```

with positive moments of inertia.

---

## 8. Quaternion Kinematics

Orientation is represented by a unit quaternion.

For the chosen convention,

```text
q_dot = 0.5 * q ⊗ [0, omega]
```

where

```text
omega
```

is the body-frame angular velocity and `⊗` denotes quaternion multiplication.

The quaternion describes accumulated orientation, while angular velocity describes the instantaneous rate of rotation.

Numerical integration may cause small norm drift, so the orientation is constrained back to unit norm after integration.

Euler angles are not used as the dynamic orientation state because they introduce representation singularities.

---

## 9. Motor Forces and Torques

Each rotor produces thrust parallel to the positive body `z` axis.

The four rotor thrusts jointly determine two different effects:

```text
collective thrust
    -> translation

thrust differences + rotor reaction torques
    -> rotation
```

Total thrust is

```text
T = f1 + f2 + f3 + f4
```

Roll and pitch torques arise from rotor thrust acting at a lever arm relative to the center of mass.

Yaw torque arises from the reaction torque associated with rotor rotation.

The exact mapping between

```text
(f1, f2, f3, f4)
```

and

```text
(T, tau_x, tau_y, tau_z)
```

is fixed in `conventions.md`.

Conceptually,

```text
[f1, f2, f3, f4]
          |
          +----> total thrust
          |
          +----> roll torque
          |
          +----> pitch torque
          |
          +----> yaw torque
```

---

## 10. Full Continuous-Time Dynamics

The complete state is

```text
x = (p, v, q, omega)
```

and the action is

```text
u = (f1, f2, f3, f4)
```

The system can be written compactly as

```text
x_dot = f(x, u)
```

with

```text
p_dot = v

v_dot =
    g_W
    + (1/m) R_WB(q) F_B
    + (1/m) F_ext,W

q_dot =
    0.5 * q ⊗ [0, omega]

omega_dot =
    J^-1 [
        tau
        - omega x (J omega)
        + tau_ext
    ]
```

where `F_B` and `tau` are determined by the four rotor thrusts.

---

## 11. Causal Structure of the Dynamics

The four motor thrusts affect translation and rotation in parallel.

```text
f1..f4
  |
  +-------------------------------+
  |                               |
  v                               v
total thrust                    torques
  |                               |
  v                               v
body-frame force             angular acceleration
  |                               |
  v                               v
rotate using current q       angular velocity
  |                               |
  v                               v
world-frame force             orientation
  |                               |
  v                               |
linear acceleration <-------------+
  |
  v
velocity
  |
  v
position
```

At a given instant, all derivatives are computed from the **same current state** and current action.

Translation does not wait for a fully updated future orientation.

The sequence is

```text
x(t), u(t)
    |
    v
compute p_dot, v_dot, q_dot, omega_dot
    |
    v
integrate all derivatives
    |
    v
x(t + dt)
```

---

## 12. Numerical Time Evolution

The rigid-body equations are continuous in time, but the simulator produces a discrete state sequence

```text
x_0, x_1, x_2, ...
```

at a fixed timestep

```text
dt
```

Forward Euler is sufficient as an initial transparent reference:

```text
x_(k+1) = x_k + dt * f(x_k, u_k)
```

A fixed-step classical RK4 method is the intended more accurate integrator after the dynamics are validated.

The conceptual model remains unchanged when the numerical integration scheme changes.

Simulation time is distinct from wall-clock time. Training should normally execute faster than real time, while interactive visualization may intentionally run at approximately real-time speed.

---

## 13. Hovering Task

The target is a desired position setpoint

```text
p*
```

with stable hovering meaning approximately

```text
p -> p*
v -> 0
omega -> 0
```

while maintaining a physically valid attitude and bounded rotor thrusts.

For the nominal hover task, yaw does not need to track a specific heading unless a heading target is explicitly introduced.

The simplest task can therefore reward:

```text
small position error
small velocity
small angular velocity
reasonable orientation
reasonable control effort
```

The controller should also recover from small randomized deviations rather than succeeding only when initialized exactly at equilibrium.

---

## 14. Reinforcement-Learning Formulation

The simulator defines an environment.

At each control step:

```text
1. the policy observes the current state and hover target;
2. the policy produces four rotor thrusts;
3. the simulator evolves the quadrotor;
4. the environment assigns a scalar reward;
5. the process repeats.
```

A trajectory is

```text
x_0, u_0, r_0,
x_1, u_1, r_1,
...
```

The policy is parameterized by neural-network parameters

```text
theta
```

and represents

```text
u_t ~ pi_theta(. | observation_t)
```

during stochastic training.

The learning objective is to maximize expected cumulative reward.

The policy is still a **closed-loop feedback controller** because its action depends on the observed current state.

---

## 15. Observation

The first version uses the **true simulated state directly**.

A conceptually sufficient observation contains quantities such as

```text
position error: p - p*
linear velocity: v
orientation: q
angular velocity: omega
```

or an equivalent representation containing the same information.

The environment does not initially simulate

```text
IMU measurements
GPS measurements
sensor noise
state estimation
communication delays
```

This isolates the rigid-body control and RL problem from perception and estimation.

---

## 16. Reward

The reward should express the desired behavior without explicitly encoding a classical controller.

A generic hover objective can penalize

```text
position error
velocity magnitude
angular velocity magnitude
excessive attitude deviation
unnecessary control effort
```

Conceptually,

```text
r =
    - w_p     ||p - p*||^2
    - w_v     ||v||^2
    - w_omega ||omega||^2
    - w_q     attitude_error
    - w_u     control_cost
```

with nonnegative weights.

The reward should not prescribe explicit PX4-style rules such as

```text
if x error is positive, command a particular pitch action
```

because the purpose is for the policy itself to discover a successful feedback mapping.

Reward terms must also avoid conflicting objectives. For example, penalizing all tilt too aggressively would prevent the vehicle from tilting when horizontal correction is physically necessary.

---

## 17. Initial-State Randomization

Training should not always start from the exact hover equilibrium.

Episodes should begin with small randomized perturbations in some or all of

```text
position
linear velocity
orientation
angular velocity
```

so that the learned policy must perform recovery and stabilization.

The initial range should be physically moderate at first and can be expanded later if training remains stable.

---

## 18. Episode Termination

An episode ends when either

```text
the fixed simulation horizon is reached
```

or the state becomes clearly invalid or unrecoverable.

Possible failure conditions include

```text
position leaving a large permitted region
excessive tilt
excessive angular rate
invalid numerical state
```

Termination criteria exist to keep training focused on the region of state space relevant to stable flight.

---

## 19. Disturbances

Nominal hovering should work before disturbances are introduced.

Later experiments can add disturbances such as

```text
constant external force
random force
time-varying force
external torque
```

The disturbance belongs to the environment dynamics.

The policy does not need to observe the disturbance force directly.

Instead,

```text
disturbance
    -> changes state trajectory
    -> policy observes changed state
    -> policy reacts through feedback
```

This permits robustness experiments without changing the basic control architecture.

---

## 20. Conventional PX4 Reference

The conventional multicopter-control pipeline can be summarized as

```text
position error
    |
    v
position P
    |
    v
desired velocity
    |
    v
velocity PID
    |
    v
desired acceleration
    |
    v
desired thrust + attitude
    |
    v
attitude controller
    |
    v
desired angular rate
    |
    v
rate PID
    |
    v
desired torque
    |
    v
control allocation
    |
    v
motor commands
```

The desired acceleration determines the required thrust-vector direction and magnitude.

Ignoring disturbances,

```text
F* = m (a* - g_W)
```

so

```text
T* = ||F*||
```

and

```text
z_B* = F* / ||F*||
```

fixes the desired body thrust axis.

Acceleration determines three constraints: thrust magnitude plus two tilt degrees of freedom.

It does not determine yaw, because rotation around the thrust axis leaves the thrust vector unchanged. A desired heading must therefore be specified separately if yaw control is required.

This cascade is studied to understand the engineered solution to the same physical control problem.

---

## 21. Difference Between PX4 and the Pure-RL Controller

PX4 explicitly constructs intermediate variables:

```text
desired velocity
desired acceleration
desired attitude
desired angular rate
desired torque
```

The pure-RL policy does not need to expose any of these quantities.

Its mapping is simply

```text
(current state, target)
        |
        v
      policy
        |
        v
(f1, f2, f3, f4)
```

Both systems are closed-loop controllers.

The fundamental difference is how the feedback law is obtained:

```text
PX4:
human-designed model-based cascade

pure RL:
control law learned from interaction and reward
```

The RL policy may learn behavior that resembles parts of a classical cascade, but that structure is not explicitly imposed.

---

## 22. Validation Principles

The simulator should satisfy physically meaningful checks before reinforcement learning is trusted.

Important conceptual checks include:

```text
zero thrust
    -> free fall

upright orientation + total thrust equal to m g
    -> zero vertical acceleration

equal rotor thrusts
    -> zero roll/pitch/yaw torque

left-right thrust imbalance
    -> roll torque with the expected sign

front-rear thrust imbalance
    -> pitch torque with the expected sign

CW/CCW reaction-torque imbalance
    -> yaw torque with the expected sign
```

Numerical consistency should also be checked by comparing trajectories at decreasing timesteps and by comparing Euler against RK4.

The purpose of these checks is to validate the environment independently of the learned controller.

---

## 23. Evaluation

A trained policy should be evaluated separately from training.

The evaluation should demonstrate at least:

```text
stable nominal hover
recovery from small initial perturbations
bounded actuator outputs
position convergence
low residual velocity
low residual angular rate
```

Required plots include

```text
x setpoint vs x response
y setpoint vs y response
z setpoint vs z response

f1 over time
f2 over time
f3 over time
f4 over time
```

A 3D visualization should show the vehicle orientation, position, and trajectory over time.

Optional robustness experiments can compare performance under external disturbances.

---

## 24. Key Simplifications

The first version intentionally excludes:

```text
sensor simulation
state estimation
motor electrical dynamics
aerodynamic blade-element models
ground contact
collision detection
terrain
communication delays
ROS2
general-purpose robotics middleware
full PX4 implementation
```

The environment is designed to isolate the core problem:

```text
rigid-body quadrotor dynamics
+
direct motor control
+
reinforcement learning
```

These simplifications are deliberate and can be relaxed later without changing the conceptual foundation.

---

## 25. Final Conceptual Summary

The project can be reduced to one closed-loop equation sequence:

```text
observation_t
    |
    v
pi_theta
    |
    v
u_t = [f1, f2, f3, f4]
    |
    v
x_dot_t = f(x_t, u_t)
    |
    v
numerical integration
    |
    v
x_(t+1)
    |
    +----> next observation
```

The simulator provides the physical world.

The RL policy provides the feedback control law.

The reward defines what successful hovering means.

The numerical integrator advances the continuous dynamics through simulated time.

No implementation structure is specified in this document. Code organization, file layout, development order, testing sequence, and iteration boundaries belong exclusively in the implementation plan.
