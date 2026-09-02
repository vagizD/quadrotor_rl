import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('docs/media', exist_ok=True)

hover_move_dir = 'runs/ppo_hover_and_move/20260901_230326_seed42_246d3c56'
baseline_dir = 'runs/ppo_hover_baseline/20260831_031226_seed42_4c8d9562'

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def create_rotor_circle_body(radius=0.06, num_points=16):
    theta = np.linspace(0, 2*np.pi, num_points)
    circle = np.zeros((num_points, 3))
    circle[:, 0] = radius * np.cos(theta)
    circle[:, 1] = radius * np.sin(theta)
    return circle

# Drone physical geometry (in meters)
arm_length = 0.2
L = arm_length * np.sqrt(2)/2
motor_pos_body = np.array([
    [L, L, 0.0],    # Front-Right (M1)
    [-L, -L, 0.0],  # Rear-Left (M2)
    [L, -L, 0.0],   # Front-Left (M3)
    [-L, L, 0.0]    # Rear-Right (M4)
])
rotor_circle_body = create_rotor_circle_body(radius=0.06)

# Body frame axes unit vectors (scaled for visualization: 0.18m length)
axis_scale = 0.18
body_axis_x = np.array([axis_scale, 0.0, 0.0])  # Body X (Forward) -> Red
body_axis_y = np.array([0.0, axis_scale, 0.0])  # Body Y (Right)   -> Green
body_axis_z = np.array([0.0, 0.0, axis_scale])  # Body Z (Up)      -> Blue

# -------------------------------------------------------------
# 1. Single GIF: Far Target Trajectory (With Body Frame Axes)
# -------------------------------------------------------------
def make_far_target_gif():
    print("Generating docs/media/far_target_trajectory.gif...")
    data = np.load(f'{hover_move_dir}/evaluation/hover_scenarios/far_target.npz')
    states = data['states']

    stride = 4  # 1000 steps / 4 = 250 frames (~12.5 seconds at 20fps)
    states_sub = states[::stride]
    x, y, z = states_sub[:, 0], states_sub[:, 1], states_sub[:, 2]
    quats_wxyz = states_sub[:, 3:7]
    rot_mats = [quat_to_mat(q) for q in quats_wxyz]

    fig = plt.figure(figsize=(9, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([3.0, 3.0, 3.0])
    ax.view_init(elev=28, azim=-62)

    # Static elements
    ax.scatter([0], [0], [0], color='#D4AF37', s=180, marker='*', label='Target (0,0,0)', zorder=10)
    ax.scatter([x[0]], [y[0]], [z[0]], color='#2ECC71', s=80, label='Start Position', zorder=10)

    # Dynamic elements
    traj_line, = ax.plot([], [], [], color='#1F77B4', linewidth=2.5, alpha=0.85, label='Drone Path')
    drone_arm1, = ax.plot([], [], [], color='#2C3E50', linewidth=3)
    drone_arm2, = ax.plot([], [], [], color='#2C3E50', linewidth=3)
    hub_marker = ax.scatter([], [], [], color='#2C3E50', s=35, zorder=8)

    # Body frame axes (RGB = XYZ)
    axis_x_line, = ax.plot([], [], [], color='#E74C3C', linewidth=2.5, label='Body X (Forward)')
    axis_y_line, = ax.plot([], [], [], color='#2ECC71', linewidth=2.5, label='Body Y (Right)')
    axis_z_line, = ax.plot([], [], [], color='#3498DB', linewidth=2.5, label='Body Z (Up)')

    rotor_lines = [ax.plot([], [], [], color='#7F8C8D', linewidth=1.5)[0] for _ in range(4)]

    ax.set_title('Hover & Move Policy: Far Target Trajectory (2.0m)', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('X (m)', labelpad=8)
    ax.set_ylabel('Y (m)', labelpad=8)
    ax.set_zlabel('Z (m)', labelpad=8)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)

    def update(frame):
        traj_line.set_data(x[:frame+1], y[:frame+1])
        traj_line.set_3d_properties(z[:frame+1])

        pos = np.array([x[frame], y[frame], z[frame]])
        R_mat = rot_mats[frame]

        motors_world = motor_pos_body.dot(R_mat.T) + pos

        drone_arm1.set_data([motors_world[0, 0], motors_world[1, 0]], [motors_world[0, 1], motors_world[1, 1]])
        drone_arm1.set_3d_properties([motors_world[0, 2], motors_world[1, 2]])
        drone_arm2.set_data([motors_world[2, 0], motors_world[3, 0]], [motors_world[2, 1], motors_world[3, 1]])
        drone_arm2.set_3d_properties([motors_world[2, 2], motors_world[3, 2]])

        hub_marker._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # Body axes
        ax_x_world = pos + R_mat.dot(body_axis_x)
        ax_y_world = pos + R_mat.dot(body_axis_y)
        ax_z_world = pos + R_mat.dot(body_axis_z)

        axis_x_line.set_data([pos[0], ax_x_world[0]], [pos[1], ax_x_world[1]])
        axis_x_line.set_3d_properties([pos[2], ax_x_world[2]])
        axis_y_line.set_data([pos[0], ax_y_world[0]], [pos[1], ax_y_world[1]])
        axis_y_line.set_3d_properties([pos[2], ax_y_world[2]])
        axis_z_line.set_data([pos[0], ax_z_world[0]], [pos[1], ax_z_world[1]])
        axis_z_line.set_3d_properties([pos[2], ax_z_world[2]])

        for i in range(4):
            r_world = rotor_circle_body.dot(R_mat.T) + motors_world[i]
            rotor_lines[i].set_data(r_world[:, 0], r_world[:, 1])
            rotor_lines[i].set_3d_properties(r_world[:, 2])

        return [traj_line, drone_arm1, drone_arm2, hub_marker, axis_x_line, axis_y_line, axis_z_line] + rotor_lines

    ani = animation.FuncAnimation(fig, update, frames=len(states_sub), interval=50, blit=False)
    ani.save('docs/media/far_target_trajectory.gif', writer='pillow', fps=20, dpi=110)
    plt.close()
    print("Done far_target_trajectory.gif")


# -------------------------------------------------------------
# 2. Multi-scenario GIFs (Hover+Move & Baseline with Body Frame)
# -------------------------------------------------------------
def make_scenarios_gif(run_dir, output_path, title_prefix):
    print(f"Generating {output_path}...")
    scenarios = ['at_target', 'near_target', 'far_target']
    titles = ['At Target (0.0m)', 'Near Target (0.25m)', 'Far Target (2.0m)']

    scenario_data = []
    max_frames = 0

    for scen in scenarios:
        filepath = f'{run_dir}/evaluation/hover_scenarios/{scen}.npz'
        if os.path.exists(filepath):
            d = np.load(filepath)
            states = d['states']
            stride = 6
            sub = states[::stride]
            quats = sub[:, 3:7]
            rmats = [quat_to_mat(q) for q in quats]
            scenario_data.append({
                'x': sub[:, 0], 'y': sub[:, 1], 'z': sub[:, 2],
                'rmats': rmats, 'len': len(sub), 'exists': True
            })
            max_frames = max(max_frames, len(sub))
        else:
            scenario_data.append({'exists': False, 'len': 0})

    fig = plt.figure(figsize=(16, 5.8), facecolor='white')
    axs = []

    traj_lines = []
    arm1_lines = []
    arm2_lines = []
    hub_markers = []
    ax_x_lines, ax_y_lines, ax_z_lines = [], [], []
    rotor_line_groups = []

    for i, (scen_info, title) in enumerate(zip(scenario_data, titles)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('X (m)', labelpad=4)
        ax.set_ylabel('Y (m)', labelpad=4)
        ax.set_zlabel('Z (m)', labelpad=4)
        ax.view_init(elev=28, azim=-62)

        if not scen_info['exists']:
            ax.text2D(0.5, 0.5, "Scenario Not Found", transform=ax.transAxes, ha='center')
            axs.append(ax)
            continue

        x, y, z = scen_info['x'], scen_info['y'], scen_info['z']
        ax.set_xlim([min(-0.5, np.min(x)-0.3), max(2.5, np.max(x)+0.3)])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_box_aspect([2.5, 2.5, 2.5])

        ax.scatter([0], [0], [0], color='#D4AF37', s=140, marker='*', zorder=10)
        ax.scatter([x[0]], [y[0]], [z[0]], color='#2ECC71', s=60, zorder=10)

        t_line, = ax.plot([], [], [], color='#1F77B4', linewidth=2, alpha=0.85)
        a1, = ax.plot([], [], [], color='#2C3E50', linewidth=2.5)
        a2, = ax.plot([], [], [], color='#2C3E50', linewidth=2.5)
        h_mark = ax.scatter([], [], [], color='#2C3E50', s=25, zorder=8)

        xx_line, = ax.plot([], [], [], color='#E74C3C', linewidth=2)
        xy_line, = ax.plot([], [], [], color='#2ECC71', linewidth=2)
        xz_line, = ax.plot([], [], [], color='#3498DB', linewidth=2)

        r_group = [ax.plot([], [], [], color='#7F8C8D', linewidth=1.2)[0] for _ in range(4)]

        traj_lines.append(t_line)
        arm1_lines.append(a1)
        arm2_lines.append(a2)
        hub_markers.append(h_mark)
        ax_x_lines.append(xx_line)
        ax_y_lines.append(xy_line)
        ax_z_lines.append(xz_line)
        rotor_line_groups.append(r_group)
        axs.append(ax)

    fig.suptitle(title_prefix, fontsize=15, fontweight='bold', y=0.97)
    plt.subplots_adjust(top=0.83, bottom=0.08, left=0.04, right=0.96, wspace=0.25)

    def update(frame):
        artists = []
        for i, scen_info in enumerate(scenario_data):
            if not scen_info['exists']:
                continue
            idx = min(frame, scen_info['len'] - 1)
            x, y, z = scen_info['x'], scen_info['y'], scen_info['z']
            R_mat = scen_info['rmats'][idx]
            pos = np.array([x[idx], y[idx], z[idx]])

            traj_lines[i].set_data(x[:idx+1], y[:idx+1])
            traj_lines[i].set_3d_properties(z[:idx+1])

            motors_world = motor_pos_body.dot(R_mat.T) + pos
            arm1_lines[i].set_data([motors_world[0, 0], motors_world[1, 0]], [motors_world[0, 1], motors_world[1, 1]])
            arm1_lines[i].set_3d_properties([motors_world[0, 2], motors_world[1, 2]])
            arm2_lines[i].set_data([motors_world[2, 0], motors_world[3, 0]], [motors_world[2, 1], motors_world[3, 1]])
            arm2_lines[i].set_3d_properties([motors_world[2, 2], motors_world[3, 2]])

            hub_markers[i]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

            # Body axes
            ax_x_w = pos + R_mat.dot(body_axis_x)
            ax_y_w = pos + R_mat.dot(body_axis_y)
            ax_z_w = pos + R_mat.dot(body_axis_z)

            ax_x_lines[i].set_data([pos[0], ax_x_w[0]], [pos[1], ax_x_w[1]])
            ax_x_lines[i].set_3d_properties([pos[2], ax_x_w[2]])
            ax_y_lines[i].set_data([pos[0], ax_y_w[0]], [pos[1], ax_y_w[1]])
            ax_y_lines[i].set_3d_properties([pos[2], ax_y_w[2]])
            ax_z_lines[i].set_data([pos[0], ax_z_w[0]], [pos[1], ax_z_w[1]])
            ax_z_lines[i].set_3d_properties([pos[2], ax_z_w[2]])

            for r in range(4):
                r_world = rotor_circle_body.dot(R_mat.T) + motors_world[r]
                rotor_line_groups[i][r].set_data(r_world[:, 0], r_world[:, 1])
                rotor_line_groups[i][r].set_3d_properties(r_world[:, 2])

            artists.extend([traj_lines[i], arm1_lines[i], arm2_lines[i], hub_markers[i],
                            ax_x_lines[i], ax_y_lines[i], ax_z_lines[i]] + rotor_line_groups[i])

        return artists

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50, blit=False)
    ani.save(output_path, writer='pillow', fps=20, dpi=100)
    plt.close()
    print(f"Done {output_path}")

if __name__ == '__main__':
    make_far_target_gif()
    make_scenarios_gif(hover_move_dir, 'docs/media/hover_move_demos.gif', 'Hover & Move Policy Evaluation Scenarios')
    make_scenarios_gif(baseline_dir, 'docs/media/baseline_demos.gif', 'Baseline Policy Evaluation Scenarios')
