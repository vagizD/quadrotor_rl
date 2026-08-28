# Implementation Plan — Pure-RL Quadrotor Hovering

## 1. Purpose

This document defines how the project should be implemented.

It is intentionally different from:

```text
task.md          -> what must be delivered
description.md   -> how the system works mathematically/conceptually
conventions.md   -> frozen frames, signs, quaternion, and motor conventions
styleguide.md    -> coding rules
```

This file defines:

```text
directory structure
module responsibilities
development sequence
test sequence
experiment/run/checkpoint structure
Codex iteration boundaries
```

## 2. Codex Working Rule

For every iteration:

- modify at most about 50 lines of code;
- solve one narrow problem only;
- do not anticipate later architecture unless required now;
- explain the mathematical meaning of the code added;
- run or provide the exact test/check for the change;
- stop after the iteration and wait for review.

Each Codex iteration should report:

```text
Goal
Files changed
What was implemented
Why it is correct
How to test it
What comes next
```

The user should be able to understand and, in principle, rewrite every iteration manually.

## 3. Architectural Principle

Split the project by real conceptual responsibility.

```text
geometry
    -> reusable rotation mathematics

simulation
    -> reusable numerical integration

robots/quadrotor
    -> quadrotor-specific state, parameters, actuators, dynamics

environments
    -> task definitions built on robot physics

rl/<algorithm>
    -> algorithm-specific learning logic

experiments
    -> configuration loading, run creation, reproducibility

visualization
    -> plotting and 3D rendering
```

Avoid creating abstract base classes merely because multiple implementations may exist later.

## 4. Target Project Structure

```text
project_root/
|
|-- task.md
|-- description.md
|-- conventions.md
|-- styleguide.md
|-- implementation_plan.md
|
|-- src/
|   |-- geometry/
|   |   |-- __init__.py
|   |   |-- rotation.py
|   |
|   |-- simulation/
|   |   |-- __init__.py
|   |   |-- integrators.py
|   |
|   |-- robots/
|   |   |-- __init__.py
|   |   |-- quadrotor/
|   |       |-- __init__.py
|   |       |-- state.py
|   |       |-- params.py
|   |       |-- motors.py
|   |       |-- dynamics.py
|   |
|   |-- environments/
|   |   |-- __init__.py
|   |   |-- quadrotor/
|   |       |-- __init__.py
|   |       |-- hover.py
|   |
|   |-- rl/
|   |   |-- __init__.py
|   |   |-- ppo/
|   |       |-- __init__.py
|   |       |-- policy.py
|   |       |-- rollout.py
|   |       |-- trainer.py
|   |
|   |-- experiments/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- runner.py
|   |   |-- tracking.py
|   |
|   |-- visualization/
|       |-- __init__.py
|       |-- plotting.py
|       |-- viewer.py
|
|-- scripts/
|   |-- train.py
|   |-- evaluate.py
|   |-- visualize.py
|
|-- configs/
|   |-- ppo_hover_baseline.toml
|
|-- tests/
|   |-- test_rotation.py
|   |-- test_motors.py
|   |-- test_dynamics.py
|   |-- test_integrators.py
|   |-- test_quadrotor_hover_environment.py
|
|-- runs/
    |-- <experiment_name>/
        |-- <run_id>/
            |-- config.toml
            |-- metadata.json
            |-- metrics.csv
            |-- checkpoints/
            |-- plots/
            |-- evaluation/
```

Do not create all files immediately. Create files only when their implementation step begins.

## 5. Module Responsibilities

### `geometry/rotation.py`

Reusable quaternion and rotation mathematics:

```text
quaternion normalization
quaternion multiplication
quaternion -> rotation matrix
small rotation helpers if needed
```

This module must not know anything about quadrotors.

### `simulation/integrators.py`

Reusable numerical integration:

```text
Euler
fixed-step classical RK4
```

The integrator knows how to advance a dynamical system but does not know what robot is being simulated.

### `robots/quadrotor/state.py`

Defines the quadrotor state semantics:

```text
position
velocity
quaternion
angular velocity
```

### `robots/quadrotor/params.py`

Defines the meaning and validation of quadrotor physical parameters:

```text
mass
arm length
inertia
yaw reaction-torque coefficient
rotor thrust bounds
gravity magnitude if treated as configurable physics
```

Run-specific parameter values come from experiment configuration.

### `robots/quadrotor/motors.py`

Maps:

```text
(f1, f2, f3, f4)
    ->
(total thrust, tau_x, tau_y, tau_z)
```

All signs must match `conventions.md`.

### `robots/quadrotor/dynamics.py`

Defines:

```text
(state, motor thrusts, physical parameters, disturbances)
    ->
state derivative
```

It contains physics only.

It must not:

```text
integrate time
choose actions
compute reward
define episodes
perform RL logic
```

### `environments/quadrotor/hover.py`

Defines the quadrotor hover task on top of quadrotor physics.

Environments remain separate from robot physics, but they are namespaced by robot because
their observations, actions, reset states, and task semantics depend on that robot.

For example:

```text
environments/
    quadrotor/
        hover.py
        landing.py            # possible future task
    manipulator/
        reaching.py           # possible future robot/task
```

It owns task-level concepts such as:

```text
target position
reset distribution
observation
reward
termination
episode horizon
task-specific disturbance schedule
```

It connects the robot dynamics and numerical integrator into an RL-style step/reset interface.

The environment therefore knows which robot it is wrapping and what that robot's action/state
semantics are, but it does not own or redefine the robot's equations of motion.

Changing the quadrotor actuator layout (for example, four rotors to six rotors) changes the
robot action model and therefore requires a matching environment action interface. The task
concept of "hover" can remain the same even though the action dimension changes.

The environment does not implement PPO.

### `rl/ppo/policy.py`

Contains PPO actor/critic neural-network definitions.

### `rl/ppo/rollout.py`

Contains trajectory collection data and rollout logic.

### `rl/ppo/trainer.py`

Contains PPO-specific learning logic:

```text
returns
advantages
clipped policy objective
value loss
entropy term if used
optimizer update
```

Other algorithms should live in their own subpackages, for example:

```text
rl/sac/
rl/td3/
```

Do not create them until actually needed.

### `experiments/config.py`

Loads and validates experiment configuration.

A full experiment config should conceptually contain:

```text
[experiment]
name
seed

[quadrotor]
mass
arm_length
inertia
yaw_torque_coefficient
max_thrust

[simulation]
dt
integrator

[environment]
task parameters
episode horizon
reset ranges
disturbance settings

[reward]
reward weights

[algorithm]
name

[ppo]
PPO hyperparameters

[network]
architecture settings
```

Use TOML for human-readable configuration.

### `experiments/runner.py`

Creates run directories, freezes resolved config, records metadata, and provides the selected algorithm with the experiment configuration.

It should not contain PPO mathematics.

### `experiments/tracking.py`

Provides training observability and persistent metric logging.

Every training run must expose progress in three forms:

```text
console summaries
TensorBoard-compatible event logs
metrics.csv
```

The training process must never run silently for long periods.

At a configurable logging interval, report at least:

```text
environment steps
training updates
mean episode return
rolling mean episode return
mean episode length
RMS position error
policy loss
value loss
action statistics
throughput / simulation steps per second
```

When available, also track useful PPO diagnostics such as:

```text
entropy
approximate KL divergence
clip fraction
explained variance
```

Reward alone is not sufficient to judge learning. Physical task metrics such as RMS position
error and stability/success rate should be visible alongside reward.

TensorBoard is the primary live visualization during training; `metrics.csv` is the durable,
tool-independent record used for later analysis and plotting.

### `visualization/plotting.py`

Produces required plots:

```text
x/y/z setpoints vs responses
f1/f2/f3/f4 vs time
training/evaluation metrics
```

### `visualization/viewer.py`

Provides 3D drone/trajectory visualization independent of physics.

### `scripts/train.py`

One top-level training entry point.

Conceptually:

```text
config path
    ->
resolved experiment config
    ->
selected algorithm implementation
    ->
run directory
```

The entry point should not require a separate script for every algorithm.

### `scripts/evaluate.py`

Loads a completed run/checkpoint and its saved configuration, reconstructs the required environment/policy, evaluates it, and stores evaluation outputs in that run.

### `scripts/visualize.py`

Visualizes a saved run or deterministic simulator trajectory.

## 6. Experiment, Run, and Checkpoint Structure

An experiment is a named configuration family, for example:

```text
ppo_hover_baseline
ppo_hover_wind
sac_hover_baseline
```

A run is one concrete execution with an exact resolved configuration and seed.

Naming:

```text
experiment:
<algorithm>_<task>_<variant>

run:
<timestamp>_seed<seed>_<config_hash>
```

Example:

```text
runs/
└── ppo_hover_baseline/
    └── 20260828_171530_seed42_a83f91/
```

Each run stores:

```text
config.toml
metadata.json
metrics.csv

checkpoints/
    step_000100000.pt
    step_000200000.pt
    best.pt
    final.pt

plots/

evaluation/
    trajectory.npz
    position_response.png
    motor_outputs.png
    metrics.json
```

The run-local `config.toml` is the fully resolved immutable configuration actually used for the run.

The metadata should include when available:

```text
experiment name
seed
Git commit hash
Python version
PyTorch version
device
start time
```

Do not encode all parameter values in filenames.

## 7. Algorithm Independence

The training entry point should use the algorithm name from configuration:

```text
[algorithm]
name = "ppo"
```

Initially, explicit dispatch is sufficient.

Do not introduce a generic `BaseAlgorithm` hierarchy before a second algorithm exists.

Later, another continuous-action algorithm such as SAC or TD3 can be added under its own package while keeping the external workflow:

```text
train <config>
evaluate <run/checkpoint>
```

stable.

Classical Q-learning/DQN is not directly interchangeable with the current continuous four-thrust action space unless the action space is discretized.

## 8. Development Phases

```text
Phase A  data and conventions
Phase B  rotation mathematics
Phase C  motor model
Phase D  translational dynamics
Phase E  rotational dynamics
Phase F  full derivative
Phase G  integration
Phase H  simulator validation
Phase I  experiment configuration foundation
Phase J  hover environment
Phase K  visualization and evaluation tools
Phase L  RL formulation freeze
Phase M  actor/critic
Phase N  rollout collection
Phase O  PPO optimization
Phase P  training, monitoring, and run management
Phase Q  evaluation
Phase R  disturbance experiments
```

RL implementation must not begin until the simulator passes nominal physical tests.

# Phase A — Data and Physical Parameters

## Iteration A1 — State container

Create the state representation only.

Checks:

```text
position shape = 3
velocity shape = 3
quaternion shape = 4
angular velocity shape = 3
```

## Iteration A2 — Physical parameter schema

Define validated physical parameters only.

Do not hard-code experiment-specific values beyond minimal defaults used for focused tests.

# Phase B — Rotation Mathematics

## Iteration B1 — Quaternion normalization

Tests:

```text
normalize([1,0,0,0]) -> identity
output norm -> 1
```

## Iteration B2 — Quaternion multiplication

Tests:

```text
identity * q = q
q * identity = q
```

## Iteration B3 — Quaternion to rotation matrix

Tests:

```text
identity quaternion -> identity matrix
R.T @ R -> I
det(R) -> +1
known 90-degree vector rotation
```

# Phase C — Motor Model

## Iteration C1 — Total thrust

Implement and test:

```text
T = f1 + f2 + f3 + f4
```

## Iteration C2 — Roll and pitch torque

Add lever-arm torques and verify exact signs from `conventions.md`.

## Iteration C3 — Yaw reaction torque

Tests:

```text
equal thrusts -> zero yaw torque
CW/CCW imbalance -> expected yaw sign
```

# Phase D — Translational Dynamics

## Iteration D1 — Position derivative

```text
p_dot = v
```

## Iteration D2 — Gravity-only acceleration

```text
zero thrust -> v_dot = [0,0,-g]
```

## Iteration D3 — Upright thrust

```text
T = mg -> zero vertical acceleration
T > mg -> positive vertical acceleration
```

## Iteration D4 — Tilted thrust

Known tilt -> expected horizontal acceleration sign.

## Iteration D5 — External force

```text
delta acceleration = F_ext / m
```

# Phase E — Rotational Dynamics

## Iteration E1 — Principal-axis torque response

With zero angular velocity:

```text
omega_dot = J^-1 tau
```

## Iteration E2 — Full rigid-body cross term

Add:

```text
omega x (J omega)
```

## Iteration E3 — Quaternion derivative

```text
q_dot = 0.5 * q ⊗ [0, omega]
```

# Phase F — Full Quadrotor Derivative

## Iteration F1 — Assemble derivative

Combine:

```text
p_dot
v_dot
q_dot
omega_dot
```

All must be evaluated from the same current state/action.

## Iteration F2 — Hover equilibrium

With:

```text
v = 0
q = identity
omega = 0
f1 = f2 = f3 = f4 = mg/4
```

expect approximately zero state derivative.

# Phase G — Numerical Integration

## Iteration G1 — Euler

Test on a trivial known ODE.

## Iteration G2 — Euler + quadrotor

Test:

```text
short hover
free fall
```

## Iteration G3 — RK4

Implement fixed-step classical RK4 without changing physics.

## Iteration G4 — Convergence comparison

Compare:

```text
dt
dt/2
dt/4
Euler
RK4
```

# Phase H — Simulator Validation

Required deterministic scenarios:

```text
free fall
hover equilibrium
vertical climb
vertical descent
pure roll
pure pitch
pure yaw
tilted thrust -> horizontal acceleration
```

No RL until these pass.

# Phase I — Experiment Configuration Foundation

## Iteration I1 — Minimal TOML config

Create one baseline experiment config containing only currently implemented physical/simulation values.

## Iteration I2 — Config loading

Load and validate the config.

## Iteration I3 — Resolved config snapshot

Create the mechanism that can serialize the exact resolved configuration.

Do not create training run infrastructure yet.

# Phase J — Hover Environment

## Iteration J1 — Reset and task state

Add:

```text
quadrotor state
hover target
episode time
reset()
```

Initially reset to nominal hover.

## Iteration J2 — Observation

Freeze exact observation scalar order.

Recommended first observation:

```text
position error
linear velocity
quaternion
angular velocity
```

## Iteration J3 — Action validation

Guarantee:

```text
0 <= f_i <= f_max
```

## Iteration J4 — Environment step

One step:

```text
receive action
advance physics
update task state/time
return observation
```

## Iteration J5 — Reward

Add initial hover objective.

## Iteration J6 — Termination

Add time horizon and clear failure bounds.

## Iteration J7 — Initial-state randomization

Add small perturbations to:

```text
position
velocity
attitude
angular velocity
```

# Phase K — Visualization and Evaluation Tools

## Iteration K1 — Position plots

Plot x/y/z target vs response.

## Iteration K2 — Motor plots

Plot f1..f4 over time.

## Iteration K3 — 3D viewer

Visualize vehicle pose, trajectory, and target.

# Phase L — RL Formulation Freeze

Before PPO code, explicitly approve:

```text
observation vector
action parameterization
reward weights
episode horizon
reset distribution
termination
gamma
GAE lambda
PPO clip
network architecture
learning rate
batching
normalization
```

Do not let Codex choose these silently.

# Phase M — Actor/Critic

## Iteration M1 — Actor

Small MLP forward pass only.

## Iteration M2 — Continuous action distribution

Sampling and log probability.

## Iteration M3 — Critic

Scalar value output.

## Iteration M4 — Combined interface

Only after individual components are understood.

# Phase N — Rollout Collection

## Iteration N1 — Single trajectory

Collect:

```text
observations
actions
rewards
values
log probabilities
done flags
```

## Iteration N2 — Returns

Verify with a hand-computable example.

## Iteration N3 — Advantages / GAE

Add only after plain returns are verified.

# Phase O — PPO Optimization

## Iteration O1 — Probability ratio

```text
ratio = exp(new_log_prob - old_log_prob)
```

## Iteration O2 — Clipped surrogate

Implement and manually verify on tiny tensors.

## Iteration O3 — Value loss

## Iteration O4 — Entropy term if needed

## Iteration O5 — Optimizer update

Verify finite loss, gradients, and parameter change.

# Phase P — Training, Monitoring, and Run Management

## Iteration P1 — One PPO update cycle

```text
collect
compute returns/advantages
update
```

Verify one complete update without a long training run.

## Iteration P2 — Run directory creation

Create:

```text
runs/<experiment>/<run_id>/
```

and save resolved config + metadata before long-running training begins.

## Iteration P3 — Persistent metric logging

Write training metrics to:

```text
metrics.csv
```

Start with a small, explicit set of metrics and verify the file contents on a short run.

## Iteration P4 — Console progress summaries

Print compact progress every configurable number of updates.

At minimum show:

```text
environment steps
update number
mean/rolling episode return
RMS position error
episode length
policy loss
value loss
simulation throughput
```

The training process must not be silent.

## Iteration P5 — TensorBoard live monitoring

Write the same core metrics to TensorBoard-compatible event logs inside the run directory.

Verify on a short run that curves update live.

Primary curves:

```text
mean and rolling episode return
RMS position error
episode length / success rate
policy loss
value loss
action mean/std
simulation throughput
```

Useful PPO diagnostics may include:

```text
entropy
approximate KL divergence
clip fraction
explained variance
```

Do not rely on reward alone; physical hover metrics must be visible.

## Iteration P6 — Periodic deterministic evaluation probe

Every configurable number of training updates, evaluate the current policy from one or more
fixed initial conditions without exploration noise.

Log at least:

```text
RMS position error
final position error
mean speed
maximum angular rate
episode success/failure
```

These evaluation probes should use the same initial conditions across training so that progress
is directly comparable. They provide a cleaner learning signal than stochastic training reward.

Optionally save/update compact position-response plots for selected evaluation checkpoints.

## Iteration P7 — Repeated updates

Enable long-running training only after console, CSV, TensorBoard, and periodic evaluation
monitoring all work.

## Iteration P8 — Checkpointing

Save:

```text
step_*.pt
best.pt
final.pt
```

A checkpoint must remain traceable to its run-local resolved config.

## Iteration P9 — Training stabilization

Only then consider:

```text
observation normalization
advantage normalization
reward scaling
gradient clipping
learning-rate tuning
```

one change at a time.

# Phase Q — Evaluation

Load the completed run and its saved configuration.

Required scenarios:

```text
nominal hover
position perturbation
velocity perturbation
attitude perturbation
```

Store evaluation data and plots inside the run.

Metrics may include:

```text
RMS position error
maximum position error
final position error
mean speed
maximum angular rate
mean control effort
```

# Phase R — Disturbances

Only after nominal hover succeeds.

Add one at a time:

```text
constant lateral force
constant vertical force
random bounded force
time-varying force
optional external torque
```

If robustness training is added, make the disturbance configuration part of the experiment config.

## 9. Minimal Success Path

Prioritize:

```text
1. correct conventions
2. correct physics
3. physical tests
4. Euler + RK4
5. reproducible config
6. hover environment
7. PPO
8. training/run management
9. x/y/z plots
10. four motor plots
11. 3D visualization
12. final explanation
```

Defer:

```text
manual keyboard control
advanced disturbances
sensor models
state estimation
ROS2
C++ rewrite
high-fidelity aerodynamics
multiple RL algorithms
```

## 10. Definition of Done

The core project is complete when:

- simulator sanity tests pass;
- RK4 produces stable nominal physics at the chosen timestep;
- the quadrotor hover environment has explicit observation, reward, reset, and termination behavior;
- PPO can stabilize near the desired hover setpoint;
- training exposes live console and TensorBoard progress plus durable CSV metrics;
- run configuration and checkpoints are reproducible;
- the policy outputs four rotor thrusts directly;
- required plots and 3D visualization are generated;
- dynamics, integration, environment, and RL algorithm remain conceptually separate;
- every implemented component is understood and reviewable by the user.
