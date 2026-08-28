# Conventions — Pure-RL Quadrotor

This file freezes the mathematical and physical conventions used throughout the project.
If code, equations, tests, or diagrams disagree with this file, this file wins.

## 1. Units

Use SI units everywhere.

```text
position            m
velocity            m/s
acceleration        m/s^2
mass                kg
force / thrust      N
torque               N*m
angular velocity    rad/s
time                 s
```

Use `g = 9.81 m/s^2` unless a test explicitly overrides it.

## 2. Coordinate Frames

Use two right-handed frames.

```text
WORLD FRAME W                 BODY FRAME B

        +z_W                         +z_B
          ^                            ^
          |                            |
          |                            |
          o ----> +x_W                o ----> +x_B  forward
         /
      +y_W                           /
                                    +y_B  left
```

World frame:

```text
+x_W : fixed world x direction
+y_W : fixed world y direction
+z_W : upward
```

Body frame:

```text
+x_B : forward
+y_B : left
+z_B : upward, along total rotor thrust
```

Gravity therefore points along `-z_W`.

```text
g_W = [0, 0, -g]
```

## 3. Right-Hand Rule

All positive rotations follow the right-hand rule.

```text
roll  phi   : rotation about +x_B
pitch theta : rotation about +y_B
yaw   psi   : rotation about +z_B
```

Positive yaw rotates `+x_B` toward `+y_B`.

Euler angles are used only for explanation, debugging, and visualization.
They are not the simulator orientation state.

## 4. Orientation Representation

Represent orientation with a unit quaternion.

```text
q_WB = [q_w, q_x, q_y, q_z]
```

Scalar-first ordering is always used.

`q_WB` represents the orientation of the body frame relative to the world frame and rotates
a vector expressed in body coordinates into world coordinates.

```text
v_W = R_WB(q_WB) v_B
```

Therefore `R_WB` means:

```text
body coordinates -> world coordinates
```

The body thrust axis expressed in world coordinates is

```text
z_B_in_W = R_WB e_3

e_3 = [0, 0, 1]
```

Remember:

```text
q and -q represent the same physical orientation.
```

Normalize the quaternion after numerical integration to compensate for floating-point drift.

## 5. Quaternion Kinematics

Angular velocity is expressed in the body frame.

```text
omega_B = [omega_x, omega_y, omega_z]
```

For the convention above,

```text
q_dot = 0.5 * q_WB ⊗ [0, omega_B]
```

where `⊗` is quaternion multiplication.

For a small body-frame rotation during `dt`,

```text
q_next = q_current ⊗ delta_q
```

Do not use quaternion addition or component-wise subtraction to represent rotation composition
or orientation error.

## 6. State Convention

The simulator state is

```text
x = (p_W, v_W, q_WB, omega_B)
```

with

```text
p_W      in R^3   position in world frame
v_W      in R^3   linear velocity in world frame
q_WB     in R^4   unit quaternion, body -> world
omega_B  in R^3   angular velocity in body frame
```

The state has 13 scalar components.

Acceleration and angular acceleration are computed derivatives, not stored state variables.

## 7. Motor Geometry

Use a `+` quadrotor configuration.

Top view, looking from `+z_B` toward the vehicle:

```text
                         +x_B / front

                              M1
                              |
                              |
             +y_B / left  M4--o--M2  right / -y_B
                              |
                              |
                              M3

                         rear / -x_B
```

Motor positions relative to the center of mass are

```text
r1 = [ +l,  0, 0]
r2 = [  0, -l, 0]
r3 = [ -l,  0, 0]
r4 = [  0, +l, 0]
```

where `l > 0` is the arm length.

Each motor produces only positive thrust along `+z_B`.

```text
F_i,B = [0, 0, f_i]

f_i >= 0
```

The RL policy action is

```text
u = [f1, f2, f3, f4]
```

in Newtons.

## 8. Rotor Spin and Yaw Convention

For the initial model:

```text
M1, M3 : clockwise
M2, M4 : counter-clockwise
```

where clockwise/counter-clockwise is viewed from `+z_B` toward the origin.

Use the body reaction-torque sign convention

```text
M1, M3 -> positive tau_z
M2, M4 -> negative tau_z
```

with positive coefficient `c_tau`.

## 9. Thrust and Torque Mapping

Total body thrust is

```text
T = f1 + f2 + f3 + f4

F_B = [0, 0, T]
```

With the geometry above,

```text
tau_x = l * (f4 - f2)
tau_y = l * (f3 - f1)
tau_z = c_tau * (f1 - f2 + f3 - f4)
```

Equivalently,

```text
[T, tau_x, tau_y, tau_z]^T = B [f1, f2, f3, f4]^T
```

with

```text
B =
[ 1        1        1        1      ]
[ 0       -l        0        l      ]
[-l        0        l        0      ]
[ c_tau  -c_tau     c_tau   -c_tau  ]
```

These signs must be verified by unit tests.

## 10. Force and Torque Frames

Use:

```text
gravity                 world frame
external force          world frame
total rotor thrust      body frame, then rotate to world
motor torque            body frame
external torque         body frame
angular velocity        body frame
inertia matrix J        body frame
```

Thus translational dynamics rotate rotor thrust using `R_WB`.

## 11. Inertia

The inertia matrix is defined about the center of mass and expressed in the body frame.

For the initial symmetric model use a diagonal matrix:

```text
J = diag(J_x, J_y, J_z)
```

with all diagonal entries positive.

## 12. Time

Use a fixed physics timestep

```text
dt > 0
t_k = k * dt
```

The simulator must distinguish simulation time from wall-clock time.

Forward Euler is the initial debugging integrator.
Fixed-step classical RK4 is the intended higher-accuracy integrator.

## 13. Naming in Mathematics and Documentation

Use these symbols consistently:

```text
p       position
v       linear velocity
q       quaternion orientation
omega   angular velocity
a       linear acceleration
tau     torque
J       inertia matrix
R_WB    body-to-world rotation matrix
f_i     thrust produced by motor i
T       total thrust
l       arm length
g       positive gravity magnitude
dt      simulation timestep
```

Do not change frame, quaternion, motor-numbering, or sign conventions locally inside an
implementation file. Any intentional convention change must update this document first.
