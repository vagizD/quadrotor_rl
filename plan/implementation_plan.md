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
|-- docs/
|   |-- diagnose_physics.md
|
|-- src/
|   |-- common/
|   |   |-- __init__.py
|   |   |-- types.py
|   |-- geometry/
|   |   |-- __init__.py
|   |   |-- rotation.py
|   |
|   |-- simulation/
|   |   |-- __init__.py
|   |   |-- trajectory.py
|   |   |-- recorder.py
|   |   |-- integrators/
|   |       |-- __init__.py
|   |       |-- _common.py
|   |       |-- euler.py
|   |       |-- rk4.py
|   |
|   |-- robots/
|   |   |-- __init__.py
|   |   |-- quadrotor/
|   |       |-- __init__.py
|   |       |-- model.py
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
|       |-- rerun_viewer.py
|
|-- scripts/
|   |-- diagnose_physics.py
|   |-- train.py
|   |-- evaluate.py
|   |-- visualize.py
|
|-- configs/
|   |-- ppo_hover_baseline.toml
|
|-- tests/
|   |-- common.py
|   |-- geometry/
|       |-- test_rotation.py
|   |-- robots/
|       |-- quadrotor/
|           |-- test_state.py
|           |-- test_params.py
|           |-- test_motors.py
|           |-- test_dynamics.py
|           |-- test_quadrotor.py
|   |-- simulation/
|       |-- integrators/
|           |-- test_integrators.py
|       |-- test_trajectory.py
|       |-- test_recorder.py
|   |-- visualization/
|       |-- test_visualization.py
|       |-- test_rerun_viewer.py
|   |-- test_quadrotor_hover_environment.py
|   |-- physics/
|       |-- test_quadrotor_scenarios.py
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

### `common/types.py`

Shared NumPy array type aliases used across the simulator. It contains no runtime physics or
task logic.

### `geometry/rotation.py`

Reusable quaternion and rotation mathematics:

```text
Quaternion value object (unit by default; `normalize=False` for raw values)
quaternion normalization
quaternion multiplication
quaternion -> rotation matrix
small rotation helpers if needed
```

This module must not know anything about quadrotors.

### `simulation/integrators/`

Reusable numerical integration:

```text
EulerIntegrator
RK4Integrator (fixed-step classical RK4)
```

Both integrators expose the same `step(state, derivative_function)` interface and own a fixed
positive timestep. Their implementations live in separate modules, with only validation shared.

The integrator knows how to advance a dynamical system but does not know what robot is being simulated.

### `simulation/trajectory.py`

Defines a backend-independent `Trajectory` record for physics and deterministic evaluation.
It aligns state samples at `t_0 ... t_N` with thrust actions applied over the `N` transitions.
It also saves and loads validated recordings in a compressed NumPy format. Visualization libraries
must not leak into this data model.

### `simulation/recorder.py`

Connects a quadrotor, a fixed thrust schedule, and a generic integrator to the `Trajectory`
record. It uses pure derivative callbacks and does not mutate the source robot.

### `robots/quadrotor/state.py`

Defines the quadrotor state semantics:

```text
position
velocity
quaternion
angular velocity
```

### `robots/quadrotor/model.py`

Provides the stateful `Quadrotor` façade. It owns parameters, state, and the current thrust
command, and delegates motor calculations to the pure functions in `motors.py`.

### `robots/quadrotor/params.py`

Defines the meaning and validation of quadrotor physical parameters:

```text
mass
arm length
inertia
yaw reaction-torque coefficient
rotor thrust bounds
motor spin directions
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

The full torque vector is computed by one pure function. Its yaw component uses the
`motor_spin_directions` stored in `QuadrotorParams`; the default signs match `conventions.md`.

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

The project deliberately provides two access paths: pure functions remain the source of truth
for testing and future integrator callbacks, while `Quadrotor` methods provide convenient access
through stored state and parameters. These are façades over one implementation, not duplicated
physics. The future integrator must use the pure derivative path so RK4 can evaluate temporary
states without mutating the live `Quadrotor`.

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
motor_spin_directions

[simulation]
dt
integrator

[dimensions]
observation_dim
action_dim

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

The `[dimensions]` section explicitly declares the observation and action interface used by the
experiment. The loader validates these declarations against the currently implemented state
layout and quadrotor actuator layout, so a changed dimension cannot silently run with an
incompatible environment or policy.

The fixed quadrotor component dimensions and derived state slices are grouped in the immutable
`QuadrotorDimensions` value object. This keeps structural model dimensions together while the
experiment config remains the explicit source of the policy/environment interface declaration.

### `experiments/runner.py`

Creates run directories, freezes resolved config, records metadata, and provides the selected algorithm with the experiment configuration.

It should not contain PPO mathematics.

### `experiments/tracking.py`

Provides training observability and persistent metric logging.
This module belongs to Phase P and is not part of the Phase I physics-diagnostics foundation.

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

Consumes trajectory data and produces physics/evaluation plots:

```text
x/y/z setpoints vs responses
linear and angular velocity vs time
quaternion components vs time when useful
f1/f2/f3/f4 vs time
```

Later RL phases may add reward and training-metric plots without changing the physics data path.

### `visualization/rerun_viewer.py`

Logs vehicle pose, body axes, motor geometry, target, and trajectory to Rerun. It consumes a
recorded trajectory plus the visualization-only arm length, and remains independent of dynamics
and numerical integration. Rerun is an optional visualization dependency.

### `scripts/diagnose_physics.py`

Runs deterministic simulator scenarios, records one trajectory, and sends the same data to
static plotting and optional Rerun recording/playback. It contains no environment or RL logic.

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
Phase I  physics diagnostics and visualization
Phase J  experiment configuration foundation
Phase K  hover environment
Phase L  RL formulation freeze
Phase M  actor/critic
Phase N  rollout collection
Phase O  PPO optimization
Phase P  training, TensorBoard, RL visualization, and run management
Phase Q  evaluation
Phase R  disturbance experiments
```

RL implementation must not begin until the simulator passes nominal physical tests and the
Phase I diagnostic trajectories can be inspected through static plots and 3D playback.

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

## Iteration C2 — Full torque mapping

Combine lever-arm roll/pitch torque and rotor reaction yaw torque in one pure `compute_torque`
function. Motor spin signs come from `QuadrotorParams` and default to the signs in
`conventions.md`.

Tests:

```text
equal thrusts -> zero yaw torque
CW/CCW imbalance -> expected yaw sign
roll/pitch lever-arm signs -> expected torque vector
```

## Iteration C3 — Quadrotor model façade

Add a stateful `Quadrotor` object that owns `QuadrotorParams`, `QuadrotorState`, and the current
thrust command. Its motor methods delegate to the pure functions already tested in `motors.py`.

The pure and stateful APIs are intentionally both retained for future integration and testing.

# Phase D — Translational Dynamics

## Iteration D1 — Position derivative

```text
p_dot = v
```

Implement the pure derivative function first and expose it through the `Quadrotor` façade
without duplicating the calculation.

## Iteration D2 — Gravity-only acceleration

```text
zero thrust -> v_dot = [0,0,-g]
```

Implement gravity as a pure world-frame acceleration function and expose it through the
`Quadrotor` façade.

## Iteration D3 — Upright thrust

```text
T = mg -> zero vertical acceleration
T > mg -> positive vertical acceleration
```

Implement linear acceleration by rotating the body-frame thrust into the world frame and adding
gravity. Validate the upright hover and climb cases here; validate tilted thrust in D4.

## Iteration D4 — Tilted thrust

Known tilt -> expected horizontal acceleration sign.

Verify that positive pitch rotates the body thrust axis toward positive world `x`.

## Iteration D5 — External force

```text
delta acceleration = F_ext / m
```

Add the optional world-frame external-force contribution to linear acceleration and verify that
the acceleration delta equals force divided by mass.

# Phase E — Rotational Dynamics

## Iteration E1 — Angular acceleration

Implement one general angular-acceleration function covering both zero and nonzero angular
velocity. With zero angular velocity:

```text
omega_dot = J^-1 tau
```

For general angular velocity, include:

```text
omega x (J omega)
```

Expose the same function through the `Quadrotor` façade; no separate zero-rate function is
needed.

## Iteration E2 — Quaternion derivative

```text
q_dot = 0.5 * q ⊗ [0, omega]
```

Implement the pure quaternion derivative using a raw pure quaternion for `[0, omega]`, and
expose it through the `Quadrotor` façade. Return the derivative as a raw `FloatVector`, not as
an orientation quaternion.

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

Return the four rates in a `QuadrotorStateDerivative` container. The quaternion rate remains a
raw four-vector because it is a derivative, not an orientation quaternion.

## Iteration F2 — Hover equilibrium

With:

```text
v = 0
q = identity
omega = 0
f1 = f2 = f3 = f4 = mg/4
```

expect approximately zero state derivative.

Verify the complete derivative through both the pure function and the `Quadrotor` façade.

# Phase G — Numerical Integration

## Iteration G1 — Euler

Implement a robot-independent `EulerIntegrator` for flat `FloatVector` states and derivative
callbacks. The class owns a fixed positive timestep and exposes `step(state, derivative_function)`.
Validate the derivative shape and test it on a trivial known ODE. `RK4Integrator` will use the
same interface in G3.

## Iteration G2 — Euler + quadrotor

Test:

```text
short hover
free fall
```

Use the pure vector derivative callback with `EulerIntegrator` and reconstruct a normalized
`QuadrotorState` after each step. The live `Quadrotor` façade must not be mutated while the
integrator evaluates the derivative.

## Iteration G3 — RK4

Implement a robot-independent `RK4Integrator` with the same fixed-`dt` and `step` interface as
`EulerIntegrator`. It must use the pure derivative callback for all four intermediate slopes
without changing the physics.

## Iteration G4 — Convergence comparison

Compare:

```text
dt
dt/2
dt/4
Euler
RK4
```

Compare Euler and RK4 on a known ODE at all three timesteps and verify decreasing error, with the
expected first-order and fourth-order convergence behavior.

# Phase H — Simulator Validation

Keep scenario-level physics checks in `tests/physics/`, separate from component unit tests and
integrator tests. Each test names a physical configuration explicitly.

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

# Phase I — Physics Diagnostics and Visualization

This phase creates physics-first diagnostics before environment or RL work. The trajectory data
model, static plots, and 3D playback must work for deterministic simulator scenarios. Do not add
reward, policy-loss, or TensorBoard concerns yet.

## Iteration I1 — Trajectory data model

Add a backend-independent `Trajectory` class with validated arrays for:

```text
times              shape (N + 1,)
states             shape (N + 1, 13)
motor thrusts      shape (N, 4)
optional targets   fixed or shape (N + 1, 3)
```

State sample `k` describes the system at `t_k`; thrust sample `k` is applied over the transition
from `t_k` to `t_(k+1)`. This alignment must remain explicit because later rewards and done flags
will also belong to transitions.

## Iteration I2 — Physics trajectory recorder

Add `record_quadrotor_trajectory` to record a trajectory from a source `Quadrotor`, fixed thrust
schedule, selected integrator, and number of transitions. Use the existing pure vector derivative
callback and do not mutate the source `Quadrotor` while evaluating temporary states.

Start with deterministic diagnostic scenarios such as:

```text
free fall
hover equilibrium
vertical climb
pure roll, pitch, and yaw inputs
tilted thrust
```

## Iteration I3 — Static physics plots

Add trajectory save/load support using a durable, non-pickle array format so generated diagnostic
recordings can be inspected again without rerunning the simulator.

Generate reusable plots from `Trajectory`:

```text
x/y/z position vs time, with target when available
linear velocity vs time
angular velocity vs time
quaternion components vs time when debugging attitude
f1/f2/f3/f4 thrust vs transition time
```

Plotting functions consume recorded data only; they must never execute or modify physics.

## Iteration I4 — Rerun 3D playback

Visualize a recorded trajectory in Rerun with simulation time as the timeline. Log at least:

```text
quadrotor position and orientation
body axes and motor-arm geometry
world axes
trajectory path
target position when present
motor thrusts as scalar time series
```

Allow file-only `.rrd` output or optional viewer spawning. The Rerun adapter must remain optional
and isolated from physics so headless tests and training can run without opening a viewer.

## Iteration I5 — Physics diagnostics entry point

Add `scripts/diagnose_physics.py` to select a deterministic scenario and integrator, record the
trajectory once, then feed that same record to static plots and Rerun. Support saving reusable
trajectory data and an Rerun recording for later inspection.

## Iteration I6 — Diagnostics verification

Test trajectory alignment and numerical values directly. Smoke-test static plot creation and the
Rerun logging adapter without requiring an interactive window. Manually inspect at least hover,
free fall, tilted thrust, and one rotational scenario before proceeding.

The same trajectory remains the source for later deterministic RL evaluation. Phase P will add
reward, return, episode, and optimization metrics alongside these physics channels rather than
reimplementing pose, thrust, and trajectory logging.

# Phase J — Experiment Configuration Foundation

## Iteration J1 — Minimal TOML config

Create one baseline experiment config containing only currently implemented physical/simulation values.

## Iteration J2 — Config loading

Load and validate the config.

## Iteration J3 — Resolved config snapshot

Create the mechanism that can serialize the exact resolved configuration.

The single experiment config owns experiment identity, algorithm selection, physical parameters,
simulation settings, hover-task limits, and reward weights. The loader is strict: changing the
current schema makes older configs invalid instead of silently applying new defaults.

Do not create training run infrastructure yet.

# Phase K — Hover Environment

## Iteration K1 — Reset and task state

Add:

```text
quadrotor state
hover target
episode time
reset()
```

Initially reset to nominal hover.

## Iteration K2 — Observation

Freeze exact observation scalar order.

Recommended first observation:

```text
position error
linear velocity
quaternion
angular velocity
```

## Iteration K3 — Action validation

Guarantee:

```text
0 <= f_i <= f_max
```

## Iteration K4 — Environment step

One step:

```text
receive action
advance physics
update task state/time
return observation
```

## Iteration K5 — Reward

Add the initial survival-oriented hover objective.

Keep the reward formula in the hover task, but source every tunable coefficient from the experiment
config. Position, motion, uprightness, heading, alive, distance-dependent survival, failure, and
control-effort terms are task choices, not PPO losses. The configured target heading fixes the yaw
degree of freedom that is not constrained by the thrust direction alone. Normalize the raw cost with
a smooth, configuration-scaled map into `[0, 0.9)` and log both raw and normalized costs so their
relationship remains visible during training.

## Iteration K6 — Termination

Add time horizon and clear failure bounds.

Return separate `terminated` and `truncated` flags. The time horizon is a truncation; physical
termination is reserved for leaving the recoverable position region or exceeding the tilt limit.
Velocity and angular rate remain reward terms so the policy can recover from fast transients.

## Iteration K7 — Initial-state randomization

Add a configurable `RandomInitializer` that samples small perturbations in:

```text
position
velocity
attitude
angular velocity
```

The initializer name and all minimum/maximum ranges belong to the experiment configuration.
The experiment seed controls its local random-number generator, so the same resolved config
reproduces the same reset sequence. External-force disturbances remain disabled for now.

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

The current hover-task formulation freezes the configured observation dimension as position error,
world linear velocity, scalar-first quaternion, and body angular velocity. The quaternion already
contains yaw, while the fixed target heading remains in the experiment configuration. The action
dimension is configured as one bounded thrust value per rotor. The reward uses a fixed per-step alive
component, a single weighted position-error term inside a smooth normalized state/control cost, and
a configurable physical-failure penalty. The position weight is increased in the baseline so target
regulation is more important than the secondary stability terms. The normalized cost is bounded
below the alive component, so
every nonterminal survival step remains positive; no separate success or truncation bonus is used.
Physical failures use a strong configured penalty, while a non-failing horizon truncation receives a
configured penalty based on its final distance to the target. Raw and normalized costs remain
available as diagnostics. The target heading is a required
environment setting, currently zero in the baseline config. Success tolerances are shared by the
environment task and deterministic evaluation. The training objective is survival near the
configured target; the stricter tolerances are used for quality evaluation.

The baseline uses `gamma = 0.999` with `dt = 0.01`, corresponding to an approximately ten-second
e-folding discount horizon. This keeps a reward at the end of the ten-second episode relevant to
earlier decisions; gamma is part of the resolved experiment configuration rather than an external
trainer setting.

# Phase M — Actor/Critic

## Iteration M1 — Actor

Implement a small actor MLP with two `Tanh` hidden layers of 64 units. Its forward pass returns
one unconstrained pre-squash mean per rotor action.

## Iteration M2 — Continuous action distribution

Use a diagonal Gaussian in the actor's pre-squash space, transform it with `tanh`, and map it to
the configured thrust bounds. Compute log probability with the corresponding change-of-variables
term instead of clipping sampled actions.

## Iteration M3 — Critic

Implement a separate two-layer 64-unit `Tanh` MLP that returns one scalar value for each
observation.

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

Represent one contiguous collected segment as `PPOTrajectory`, a specialization of the
visualization-ready `Trajectory`. It retains the aligned physical samples and adds the PPO fields
above. Collection/storage logic remains separate and owns episode boundaries or later batching;
the trajectory object itself only represents its supplied point-A-to-point-B sequence.

## Iteration N2 — Returns

Verify with a hand-computable example.

For a segment with discount factor `gamma`, compute return-to-go backward:

```text
G_t = r_t + gamma * (1 - terminated_t) * G_{t+1}
```

The final `G_T` uses `boundary_state_value`, the critic value of the first state outside the
segment, when the segment ends by truncation or rollout length. A physical terminal transition
has zero future value. Time-limit truncation does not mask the boundary state value.

## Iteration N3 — Advantages / GAE

Add only after plain returns are verified.

Compute generalized advantage estimates from the same boundary state value:

```text
delta_t = r_t + gamma * (1 - terminated_t) * V_{t+1} - V_t
A_t = delta_t + gamma * gae_lambda * (1 - terminated_t) * A_{t+1}
```

`gae_lambda = 0` gives one-step TD residuals; `gae_lambda = 1` gives return-to-go minus the
current value estimate, up to the available segment boundary.

# Phase O — PPO Optimization

## Iteration O1 — PPO objective terms

```text
ratio = exp(new_log_prob - old_log_prob)
L_policy = -mean(min(ratio * advantage,
                     clip(ratio, 1 - epsilon, 1 + epsilon) * advantage))
L_value = mean((value - return) ** 2)
L_total = L_policy + value_loss_coef * L_value - entropy_coef * entropy
```

Implement these terms in `rl/ppo/trainer.py`. The PPO settings belong to the unified `[ppo]`
configuration, including the clipping range, learning rate, value-loss coefficient, and entropy
coefficient. The baseline uses a full rollout batch; minibatches and repeated epochs are future
training-loop decisions.

## Iteration O2 — Full-batch PPO update

Implement `PPORollout` as storage for multiple `PPOTrajectory` segments. Compute returns and GAE
within each segment, concatenate only the transition arrays into a `PPOBatch`, and update the
actor and critic once from that combined batch. Return scalar diagnostics for later tracking.

## Iteration O3 — PPO loop verification

Use one focused test to verify the probability ratio, clipping behavior, finite update diagnostics,
and parameter change after one update. Do not add separate tests for every individual torch
operation.

# Phase P — Training, TensorBoard, RL Visualization, and Run Management

This is the second visualization layer. Reuse the Phase I trajectory, static plotting, and Rerun
pose/thrust channels, then add RL-specific metrics. TensorBoard begins here because reward,
episode, and optimization signals do not exist during physics-only diagnostics.

## Iteration P1 — One PPO update cycle

```text
collect segments until the configured rollout size
compute returns/advantages per segment
combine with PPORollout
update
```

Use the configured rollout step count to collect multiple short episode segments when needed,
then verify one complete update without a long training run.

## Iteration P2 — Run directory creation

Create:

```text
runs/<experiment>/<run_id>/
```

and save resolved config + metadata before long-running training begins. The run ID uses the
timestamp/seed/config-hash convention, and the initial directory contains `config.toml`,
`metadata.json`, `checkpoints/`, `plots/`, and `evaluation/`. The training loop receives this
run-directory object instead of constructing paths independently.

## Iteration P3 — Persistent metric logging

Write training metrics to:

```text
metrics.csv
```

Start with a small, explicit schema covering update count, environment steps, episode return and
length, RMS position error, policy loss, value loss, and simulation throughput. Verify the file
contents on a short run. The logger must write inside the run directory and flush each row so the
CSV remains useful if training stops unexpectedly.

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

Use the same metric names as `metrics.csv`, including both mean and rolling episode return. Missing
values may be displayed as `-` while the corresponding metric is not yet available.

## Iteration P5 — TensorBoard live monitoring

Write the same core metrics to TensorBoard-compatible event logs inside the run directory.

Use the run-local `tensorboard/` directory and the update number as the TensorBoard global step.
Flush after each logged update so curves remain visible during a live run. Verify on a short run
that scalar event files and curves are written.

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

## Iteration P6 — RL-specific Rerun enrichment

Extend Rerun logging for selected training and deterministic-evaluation episodes with:

```text
step reward and cumulative return
reward components when defined
position error and success/failure state
action/thrust statistics
episode boundaries
policy/value losses and PPO diagnostics when useful
```

Do not stream every training transition by default. Use configurable sampling or selected
episodes so visualization does not dominate training time or recording size.

The implementation uses the single `visualization.rerun_viewer.log_trajectory` API for both
`Trajectory` and `PPOTrajectory` objects. Rerun records only sampled physical data: the 3D
trajectory, target, pose, and motor thrusts. `TrackingConfig.rerun_step_stride` controls the
default sampling policy; the caller still chooses which training or evaluation episodes to
record. Rewards and PPO diagnostics belong to the TensorBoard/training metrics path and are
not duplicated in Rerun.

## Iteration P7 — Periodic deterministic evaluation probe

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

`EvaluationConfig` stores the evaluation interval, number of episodes, and initial-state seed.
`HoverSuccessConfig`, nested under the environment configuration, stores the shared success
tolerances. `DeterministicEvaluator` samples its initial states once from a separate seeded
initializer, then reuses copies of those states for every probe. It uses the actor's bounded
distribution mean, never an exploration sample, and returns the same visualization-ready
`Trajectory` representation used by physics diagnostics. Reaching the time horizon is recorded as
truncation; success additionally requires the shared final position, speed, angular-rate, tilt,
and heading tolerances. `EvaluationSummary.as_metrics` exposes the aggregate physical metrics for
CSV and TensorBoard logging.

Optionally save/update compact position-response plots for selected evaluation checkpoints.

## Iteration P8 — Repeated updates

Enable long-running training only after console, CSV, TensorBoard, and periodic evaluation
monitoring all work.

Add `rl.ppo.training.train_experiment` and `scripts/train.py`. Each configured update collects
one `PPORollout`, performs one full-batch PPO update, records training throughput and rollout
statistics, and writes the same metric row to console, CSV, and TensorBoard. At the configured
evaluation interval, deterministic evaluation metrics are added to that row and the resulting
physical trajectories are saved under the run's `evaluation/` directory. Optional Rerun files
are controlled by `tracking.record_evaluation_rerun`; no checkpointing or repeated optimizer
epochs are introduced before P9/P10.

## Iteration P9 — Checkpointing

Save:

```text
step_*.pt
best.pt
final.pt
```

A checkpoint must remain traceable to its run-local resolved config.

Save one `step_<update>.pt` checkpoint after every update, `best.pt` whenever deterministic
evaluation improves, and `final.pt` after the configured run. Each file contains actor and critic
weights, both optimizer states, update/environment-step counters, the run ID, config hash, and
the resolved configuration snapshot. The best checkpoint prioritizes evaluation success rate,
then lower RMS position error, final position error, mean speed, and maximum angular rate.

## Iteration P10 — Training stabilization

First, reuse each collected rollout for a configurable number of PPO optimizer passes. Add
`ppo.steps_per_rollout`; `rollout_steps` remains the number of environment transitions collected,
while `steps_per_rollout` controls how many actor/critic updates reuse that batch. Recompute the
probability ratio against the stored rollout log-probabilities on every pass, and aggregate the
per-pass diagnostics for console, CSV, and TensorBoard. The first pass has ratio one by
construction; later passes make PPO clipping observable.

Add the first conditioning layer in the same resolved configuration:

```text
running observation normalization with clipping
batch-wide advantage normalization
reward scaling for return/GAE targets
independent actor and critic gradient-norm clipping
```

The rollout keeps raw observations for updating the running statistics while the policy-facing
observations remain normalized. Normalization statistics are frozen across one collected rollout,
then checkpointed with the networks. Keep the existing learning-rate field as the single tuning
knob while these changes are verified; tune it only after the diagnostics are stable.

The baseline also bounds the actor's learned pre-squash log standard deviation with configured
`min_log_std` and `max_log_std` values. Training logs the effective latent policy standard deviation
separately from the empirical physical thrust standard deviation, since the latter also includes
variation in the policy mean across states.

Only after this is verified consider:

```text
hover-centered action residuals
curriculum over the initializer ranges
```

and then:

```text
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
5. trajectory recording
6. physics plots and Rerun 3D diagnostics
7. reproducible config
8. hover environment
9. PPO
10. TensorBoard and RL-specific Rerun diagnostics
11. training/run management
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
