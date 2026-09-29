# Quadrotor Residual Disturbance Learning Platform Roadmap

## Objective

Extend `quadrotor_rl` into a research platform for precision quadrotor
docking.

The goal is not generic flight robustness. The goal is:

> Learn compensation for unknown disturbances that prevent accurate
> docking and close-range positioning.

The platform should support: 1. PX4-style deterministic control. 2.
Environmental disturbance modeling. 3. Residual disturbance learning. 4.
Future sensor-based estimation and hardware adaptation.

------------------------------------------------------------------------

# Architecture

    Desired trajectory
            |
            v
    PX4-style controller
            |
            v
    Nominal control
            |
            +
    Residual disturbance model
            |
            v
    Final control command
            |
            v
    Quadrotor physics engine
            |
            v
    State

------------------------------------------------------------------------

# Phase 1 --- Simulator Foundation

Separate:

    quadrotor physics
    +
    environment effects
    +
    controller
    +
    learning modules
    +
    visualization

The drone geometry should be configurable.

Default:

``` yaml
drone_geometry:
    type: sphere
```

Future:

``` yaml
drone_geometry:
    type: box
```

The physics model must not depend on visualization geometry.

------------------------------------------------------------------------

# Environment Effects

Implement effects as independent modules:

    environment/
        ground_effect.py
        drag.py
        wind.py

The dynamics become:

\[ `\dot{x}`{=tex}=f\_{quad}(x,u)+F\_{environment} \]

------------------------------------------------------------------------

# Phase 2 --- PX4 Controller

Implement deterministic PX4-style cascade:

    position error
          |
    velocity controller
          |
    acceleration / attitude conversion
          |
    attitude controller
          |
    motor thrust commands

No learning components.

Validate:

-   hover;
-   point-to-point docking flight.

------------------------------------------------------------------------

# Phase 3 --- Evaluation Scenarios

Each scenario consists of:

1.  Smooth ideal trajectory.
2.  Environment configuration.

Do not use step commands.

Use polynomial/minimum-jerk/spline trajectories.

## Trajectory scenarios

### Hover

Fixed target.

### Point-to-point docking

Move from initial position to docking position.

### Infinity trajectory

Smooth figure-eight trajectory for continuous tracking evaluation.

------------------------------------------------------------------------

# Environment Conditions

Each trajectory is tested in order:

## 0. No effects

No:

-   ground effect;
-   air drag;
-   wind.

------------------------------------------------------------------------

## 1. Ground effect

Model:

\[ F\_{ground}=k(h)T \]

where h is distance to surface.

Clip to zero:

\[ F\_{ground}=0 \]

when:

\[ h\>D \]

------------------------------------------------------------------------

## 2. Air drag

Example:

\[ F\_{drag}=-c\|v\|v \]

------------------------------------------------------------------------

## 3. Constant wind gusts

Test:

-   small;
-   medium;
-   strong.

------------------------------------------------------------------------

## 4. Changing wind gusts

Test:

-   small;
-   medium;
-   strong.

------------------------------------------------------------------------

# Evaluation Outputs

For every method and scenario:

## RRD files

Generate visualization files.

Must support:

-   PX4;
-   residual methods;
-   future RL.

------------------------------------------------------------------------

## Metrics

Report:

-   average tracking error;
-   maximum tracking error;
-   final docking position error;
-   final velocity error.

Tracking error:

\[ e(t)=\|\|p(t)-p\_{ideal}(t)\|\| \]

------------------------------------------------------------------------

## Plots

Generate:

Position:

\[ X,Y,Z \]

Motor thrusts:

\[ T_1,T_2,T_3,T_4 \]

Include in RRD if possible, otherwise separate plots.

------------------------------------------------------------------------

## Disturbance visualization

For scenarios with effects, visualize disturbance vectors once:

-   wind force;
-   drag force;
-   ground effect force.

------------------------------------------------------------------------

# Residual Learning Interface

Residual learners share a common interface.

The simulator should not depend on the algorithm.

Example:

    ResidualModel.predict(inputs)
            |
            v
    disturbance estimate

------------------------------------------------------------------------

# Residual Features

Different residual learners may use different features.

Possible inputs:

## Ground truth state (initial)

\[ x=(p,v,q,`\omega`{=tex}) \]

------------------------------------------------------------------------

## History

Previous states and controls for temporal models.

------------------------------------------------------------------------

## Sensors (future)

Initially use ground truth state.

Future pipeline:

    true state
        |
    sensor simulation
        |
    EKF / estimation
        |
    estimated state
        |
    residual learner

The interface must allow replacing ground truth states with estimated
states later.

Intermediate processing:

-   IMU;
-   filtering;
-   EKF;
-   localization;

will be added later.

------------------------------------------------------------------------

# Future Algorithms

Compare:

## Classical residual learning

Least squares or similar methods.

## Neural-Fly style

\[ `\Delta `{=tex}F=`\phi`{=tex}(x)a \]

with online adaptation.

## Residual RL

\[ u=u\_{PX4}+`\Delta `{=tex}u \]

PX4 remains the baseline.

------------------------------------------------------------------------

# Development Order

1.  Build environment layer.
2.  Add ground effect, drag, wind.
3.  Validate physics.
4.  Implement PX4 controller.
5.  Validate scenarios.
6.  Add residual learning interface.
7.  Implement and compare algorithms.

------------------------------------------------------------------------

# Final Goal

Build:

\[ `\boxed{
PX4
+
environment disturbances
+
residual learning
+
precision docking evaluation
}`{=tex} \]

The platform should later support:

-   real flight data;
-   estimated states;
-   hardware-specific effects.
