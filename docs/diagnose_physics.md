# Physics Diagnostics

`diagnose_physics.py` generates a deterministic quadrotor trajectory, saves the recorded data,
and writes a static diagnostic plot.

## Setup

Install the core project dependencies with:

```bash
uv sync
```

Install the optional Rerun dependency only when `.rrd` playback is needed:

```bash
uv sync --extra visualization
```

## Basic usage

See all available options:

```bash
uv run python scripts/diagnose_physics.py --help
```

Run a hover diagnostic with Euler integration:

```bash
uv run python scripts/diagnose_physics.py \
  --scenario hover \
  --integrator euler \
  --duration 2.0 \
  --dt 0.01 \
  --output-dir artifacts/physics_diagnostics
```

The supported scenarios are:

```text
free_fall, hover, climb, roll, pitch, yaw, tilted_thrust
```

The supported integrators are `euler` and `rk4`.

## Time axis

Use `--time-axis` to choose the horizontal axis in the static plot:

```bash
uv run python scripts/diagnose_physics.py \
  --scenario climb \
  --integrator rk4 \
  --time-axis both \
  --output-dir artifacts/physics_diagnostics
```

Available values are:

```text
time  -> simulation time in seconds
step  -> discrete sample or transition index
both  -> time on the primary axis and step index on a secondary axis
```

State samples use indices `0 ... N`; motor thrust samples use transition indices `0 ... N-1`.

## Output files

For a scenario named `climb` using RK4, the output directory contains:

```text
climb_rk4.npz  # reusable Trajectory recording
climb_rk4.png  # static position, velocity, attitude, and thrust plots
```

The NumPy recording can be loaded later:

```python
from simulation import Trajectory

trajectory = Trajectory.load("artifacts/physics_diagnostics/climb_rk4.npz")
```

## Rerun recording

Add `--rerun` to save a 3D `.rrd` recording. Add `--spawn` to open the Rerun viewer while logging:

```bash
uv run --extra visualization python scripts/diagnose_physics.py \
  --scenario tilted_thrust \
  --integrator rk4 \
  --duration 2.0 \
  --dt 0.01 \
  --time-axis both \
  --rerun \
  --spawn \
  --output-dir artifacts/physics_diagnostics
```

Without `--spawn`, the recording is saved headlessly and can be opened later with:

```bash
uv run --extra visualization rerun artifacts/physics_diagnostics/tilted_thrust_rk4.rrd
```

Saved recordings include a default Rerun blueprint, so the command above opens a 3D view and a
motor-thrust timeline automatically. Recordings generated before this blueprint was added should
be regenerated. The view contains the world and body axes, motor geometry, trajectory path, target,
pose, orientation, and per-motor thrust time series.
