import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('docs', exist_ok=True)

# Load data
hover_move_dir = 'runs/ppo_hover_and_move/20260901_230326_seed42_246d3c56'
data = np.load(f'{hover_move_dir}/evaluation/hover_scenarios/far_target.npz')
states = data['states']

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# Subsample states for faster animation (dt=0.01s is too dense, let's take every 5th frame for 20fps)
stride = 5
states = states[::stride]
x, y, z = states[:, 0], states[:, 1], states[:, 2]
quats_wxyz = states[:, 3:7] # w, x, y, z
rot_matrices = [quat_to_mat(q) for q in quats_wxyz]

# Quadrotor geometry
arm_length = 0.2
# Let's draw an X shape
L = arm_length * np.sqrt(2)/2
motor_pos_body = np.array([
    [L, L, 0],
    [-L, -L, 0],
    [L, -L, 0],
    [-L, L, 0]
])

fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# Static elements
target = ax.scatter([0], [0], [0], color='gold', s=200, marker='*', label='Target (0,0,0)', zorder=5)
start = ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, label='Start Position', zorder=5)

# Dynamic elements
trajectory_line, = ax.plot([], [], [], color='blue', alpha=0.5, linewidth=2, label='Trajectory')
drone_arms1, = ax.plot([], [], [], color='black', linewidth=3)
drone_arms2, = ax.plot([], [], [], color='black', linewidth=3)
drone_rotors, = ax.plot([], [], [], marker='o', color='red', linestyle='None', markersize=8)

ax.set_title('Quadrotor Hover & Move: Far Target Navigation', fontsize=16)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Set limits
ax.set_xlim([min(x)-0.5, max(x)+0.5])
ax.set_ylim([min(y)-0.5, max(y)+0.5])
ax.set_zlim([min(z)-0.5, max(z)+0.5])
ax.legend()

def init():
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    drone_arms1.set_data([], [])
    drone_arms1.set_3d_properties([])
    drone_arms2.set_data([], [])
    drone_arms2.set_3d_properties([])
    drone_rotors.set_data([], [])
    drone_rotors.set_3d_properties([])
    return trajectory_line, drone_arms1, drone_arms2, drone_rotors

def update(frame):
    # Update trajectory history
    trajectory_line.set_data(x[:frame], y[:frame])
    trajectory_line.set_3d_properties(z[:frame])
    
    # Current pose
    pos = np.array([x[frame], y[frame], z[frame]])
    rot_mat = rot_matrices[frame]
    
    # Transform motor positions
    motor_pos_world = motor_pos_body.dot(rot_mat.T) + pos
    
    # Update arms
    # Arm 1 connects motor 0 and 1
    arm1_x = [motor_pos_world[0, 0], motor_pos_world[1, 0]]
    arm1_y = [motor_pos_world[0, 1], motor_pos_world[1, 1]]
    arm1_z = [motor_pos_world[0, 2], motor_pos_world[1, 2]]
    drone_arms1.set_data(arm1_x, arm1_y)
    drone_arms1.set_3d_properties(arm1_z)
    
    # Arm 2 connects motor 2 and 3
    arm2_x = [motor_pos_world[2, 0], motor_pos_world[3, 0]]
    arm2_y = [motor_pos_world[2, 1], motor_pos_world[3, 1]]
    arm2_z = [motor_pos_world[2, 2], motor_pos_world[3, 2]]
    drone_arms2.set_data(arm2_x, arm2_y)
    drone_arms2.set_3d_properties(arm2_z)
    
    # Update rotors
    drone_rotors.set_data(motor_pos_world[:, 0], motor_pos_world[:, 1])
    drone_rotors.set_3d_properties(motor_pos_world[:, 2])
    
    # Make the view follow the drone a bit or rotate
    ax.view_init(elev=20., azim=frame * (360. / len(states)))
    
    return trajectory_line, drone_arms1, drone_arms2, drone_rotors

ani = animation.FuncAnimation(fig, update, frames=len(states),
                              init_func=init, blit=False, interval=50)

# Save as gif
ani.save('docs/far_target_trajectory.gif', writer='pillow', fps=20, dpi=100)
print('GIF saved to docs/far_target_trajectory.gif')
