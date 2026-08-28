# Task — Pure-RL Quadrotor Hovering

## Objective

Build a 3D quadrotor simulation from scratch in Python and train a **pure reinforcement-learning controller** to achieve stable hovering at a desired position setpoint.

The final controller maps the current simulated state and hover target directly to the four rotor thrust commands. No classical PX4/PID controller sits underneath the learned policy.

## Original Deliverables

The project must demonstrate:

1. **UAV Hovering**
   - Stable hover at a desired 3D position setpoint.

2. **Dynamics & Control Explanation**
   - Explain the quadrotor rigid-body dynamics.
   - Explain the conventional cascaded multicopter-control framework, using PX4 as the reference architecture.
   - Clearly contrast the PX4-style cascade with the pure-RL controller used in this project.

3. **Position Response**
   - Plot the desired and actual `x`, `y`, and `z` positions over time.

4. **RL Control Output**
   - The neural policy outputs four rotor thrust commands directly.
   - Plot all four policy outputs over time.

5. **RL Explanation**
   - Explain reinforcement learning in our own words using this project as the concrete example.

## Project Scope

Implement a small purpose-built simulator rather than using MuJoCo, Isaac Sim, Gazebo, or another general-purpose physics engine.

The simulator models one rigid quadrotor in free 3D space with:

- gravity;
- four fixed rotors;
- rigid-body translation and rotation;
- optional external force/torque disturbances;
- numerical time integration.

## Key Simplifications

For the initial project:

- Use the **true simulated state** directly; do not model sensors, sensor noise, or state estimation.
- Do not implement the PX4 controller. PX4 is studied only as the conventional control reference.
- Use **pure RL control** with direct four-rotor thrust outputs.
- Do not model motor electrical dynamics initially; treat rotor thrust magnitude as the actuator quantity.
- Do not implement collisions, terrain, articulated bodies, or a general-purpose physics engine.
- Start with forward Euler integration for debugging and add fixed-step RK4 once the dynamics are verified.
- Use a fixed simulation timestep.
- Keep the simulator and controller modular enough to test independently, but avoid premature abstraction.

## Required Outputs

The finished project should provide:

- a tested 3D quadrotor physics simulator;
- a pure-RL hover policy implemented in PyTorch;
- a training entry point;
- an evaluation entry point;
- a 3D visualization of the quadrotor and its trajectory;
- optional keyboard/manual control for simulator validation if time permits;
- position-setpoint versus response plots for `x`, `y`, and `z`;
- plots of the four policy-generated rotor thrusts;
- quantitative hover-performance metrics;
- optional disturbance experiments after nominal hover works.

## Success Criteria

The minimum successful version should:

- remain numerically stable for the chosen timestep;
- reproduce basic physical sanity checks such as free fall and hover thrust;
- allow the learned policy to recover from small randomized initial position, velocity, and attitude errors;
- converge to and maintain a desired hover setpoint for a fixed evaluation horizon;
- produce all required plots and visualizations;
- make the separation between controller, dynamics, and numerical integration explicit in the code and documentation.

## High-Level Pipeline

```text
desired hover setpoint + current true state
                    |
                    v
              RL policy
                    |
                    v
            f1, f2, f3, f4
                    |
                    v
           quadrotor dynamics
                    |
                    v
              state derivative
                    |
                    v
          numerical integrator
                    |
                    v
               next state
                    |
                    +------ feedback to policy
```

Detailed equations, conventions, architecture, testing strategy, and RL formulation belong in `description.md` and `conventions.md`.
