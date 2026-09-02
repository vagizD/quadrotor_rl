import matplotlib.pyplot as plt
import numpy as np
import csv
import os

os.makedirs('docs/media', exist_ok=True)

hover_move_dir = 'runs/ppo_hover_and_move/20260901_230326_seed42_246d3c56'
baseline_metrics_path = 'runs/ppo_hover_baseline/20260831_122922_seed42_f787e5d1/metrics.csv'

def load_metrics(filepath):
    data = {'update': [], 'rolling_episode_return': [], 'mean_episode_length': [], 'evaluation_rms_position_error': []}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['update'].append(int(row['update']))
            data['rolling_episode_return'].append(float(row['rolling_episode_return']))
            data['mean_episode_length'].append(float(row['mean_episode_length']))
            if 'evaluation_rms_position_error' in row and row['evaluation_rms_position_error']:
                data['evaluation_rms_position_error'].append(float(row['evaluation_rms_position_error']))
            else:
                data['evaluation_rms_position_error'].append(np.nan)
    return {k: np.array(v) for k, v in data.items()}

def tensorboard_smooth(values, weight=0.74):
    """TensorBoard style exponential smoothing."""
    smoothed = np.zeros_like(values)
    last = values[0]
    for i, val in enumerate(values):
        if np.isnan(val):
            smoothed[i] = np.nan
        else:
            last = last * weight + (1.0 - weight) * val
            smoothed[i] = last
    return smoothed

metrics_base = load_metrics(baseline_metrics_path)
metrics_hm = load_metrics(f'{hover_move_dir}/metrics.csv')

# Plot 4: Training Comparison (with 0.74 EMA smoothing)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Rolling Episode Return
axs[0].plot(metrics_base['update'], metrics_base['rolling_episode_return'], color='tab:blue', alpha=0.25, linewidth=1.0)
axs[0].plot(metrics_base['update'], tensorboard_smooth(metrics_base['rolling_episode_return'], 0.74), label='Baseline (Smoothed)', color='tab:blue', linewidth=2.0)

axs[0].plot(metrics_hm['update'], metrics_hm['rolling_episode_return'], color='tab:orange', alpha=0.25, linewidth=1.0)
axs[0].plot(metrics_hm['update'], tensorboard_smooth(metrics_hm['rolling_episode_return'], 0.74), label='Hover+Move (Smoothed)', color='tab:orange', linewidth=2.0)

axs[0].set_title('Rolling Episode Return (EMA 0.74)', fontsize=12, fontweight='bold')
axs[0].set_xlabel('Update')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# 2. Mean Episode Length
axs[1].plot(metrics_base['update'], metrics_base['mean_episode_length'], color='tab:blue', alpha=0.25, linewidth=1.0)
axs[1].plot(metrics_base['update'], tensorboard_smooth(metrics_base['mean_episode_length'], 0.74), label='Baseline (Smoothed)', color='tab:blue', linewidth=2.0)

axs[1].plot(metrics_hm['update'], metrics_hm['mean_episode_length'], color='tab:orange', alpha=0.25, linewidth=1.0)
axs[1].plot(metrics_hm['update'], tensorboard_smooth(metrics_hm['mean_episode_length'], 0.74), label='Hover+Move (Smoothed)', color='tab:orange', linewidth=2.0)

axs[1].set_title('Mean Episode Length (EMA 0.74)', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Update')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# 3. Eval RMS Position Error
mask_base = ~np.isnan(metrics_base['evaluation_rms_position_error'])
mask_hm = ~np.isnan(metrics_hm['evaluation_rms_position_error'])

axs[2].plot(metrics_base['update'][mask_base], metrics_base['evaluation_rms_position_error'][mask_base], label='Baseline Eval', color='tab:blue', marker='o', markersize=3, alpha=0.7)
axs[2].plot(metrics_hm['update'][mask_hm], metrics_hm['evaluation_rms_position_error'][mask_hm], label='Hover+Move Eval', color='tab:orange', marker='o', markersize=3, alpha=0.7)
axs[2].set_title('Eval RMS Position Error', fontsize=12, fontweight='bold')
axs[2].set_xlabel('Update')
axs[2].set_ylim(0, 3)
axs[2].legend()
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/media/training_comparison.png', dpi=150)
plt.close()

# Plot 5: Thrusts over time
far_data = np.load(f'{hover_move_dir}/evaluation/hover_scenarios/far_target.npz')
fig, ax = plt.subplots(figsize=(10, 4))
thrusts = far_data['thrusts']
dt = 0.01
time = np.arange(len(thrusts)) * dt
for m in range(4):
    ax.plot(time, thrusts[:, m], label=f'Motor {m+1}', alpha=0.7)
ax.axhline(2.4525, color='black', linestyle='--', label='Nominal Hover (2.4525 N)')
ax.set_title('Hover & Move: Motor Thrusts (Far Target)', fontsize=12, fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Thrust (N)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/media/far_target_thrusts.png', dpi=150)
plt.close()

print("Generated updated training_comparison.png and far_target_thrusts.png")
