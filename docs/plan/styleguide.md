# Python Style Guide

This project follows a small, strict Python style guide intended for readable scientific/robotics code.

## General

- Target modern Python 3.
- Follow PEP 8 unless this file says otherwise.
- Prefer simple, explicit code over abstraction.
- Do not introduce inheritance, base classes, registries, factories, or generic frameworks unless the current implementation genuinely requires them.
- Prefer concrete implementations first. Generalize only after at least two real implementations need the same interface.
- Keep functions and methods focused and short.

## Naming

Use standard Python naming:

```text
classes              PascalCase
functions            snake_case
methods              snake_case
variables            snake_case
module constants     UPPER_SNAKE_CASE
private helpers       _leading_underscore
```

Examples:

```text
QuadrotorState
QuadrotorParams
reward_sum
angular_velocity
compute_derivative
GRAVITY
```

Do not use C++-style names such as `TState`, `rewardSum`, or Hungarian notation.

## Typing

- Type all function parameters and return values.
- Type dataclass fields.
- Prefer precise types over `Any`.
- Use `numpy.typing.NDArray` when useful for NumPy arrays.
- Use `torch.Tensor` for PyTorch tensors.
- Avoid complex generic typing unless it materially improves correctness.

## Data Structures

- Use `dataclass` for semantic data containers such as state and physical parameters.
- Use NumPy arrays for numerical simulation math.
- Use PyTorch tensors only inside RL/model/training code.
- Do not hide units, frames, or conventions inside ambiguous field names.

## Formatting

- Use 4 spaces for indentation.
- Use two blank lines between top-level functions/classes.
- Use one blank line between methods when it improves readability.
- Keep lines reasonably short; prefer <= 88 characters where practical.
- Use trailing commas in multiline calls/containers.
- Prefer explicit intermediate variables over dense one-line expressions when physics is involved.

## Documentation

Use Google-style docstrings for public classes/functions when behavior is not obvious.

Docstrings should explain:

- purpose;
- units and coordinate frame where relevant;
- parameter meaning;
- return meaning;
- important assumptions.

Do not restate obvious implementation details.

## Assertions and Errors

- Validate physical invariants at boundaries where useful.
- Raise clear exceptions for invalid parameters.
- Prefer assertions in tests, not as the main runtime validation mechanism.
- Never silently normalize or clip a value unless that behavior is part of the documented model.

## Testing

- Every physics equation added should have at least one focused test.
- Prefer small deterministic tests.
- Test signs, units, invariants, and known equilibria.
- Do not add RL training before the simulator passes physical sanity tests.

## Experiments and Reproducibility

Experiment names use:

```text
<algorithm>_<task>_<variant>
```

Examples:

```text
ppo_hover_baseline
ppo_hover_wind
ppo_hover_wide_reset
sac_hover_baseline
```

Rules:

- use lowercase `snake_case`;
- keep names short and human-readable;
- do not encode every hyperparameter into the experiment name;
- every run must save the fully resolved configuration used to create it;
- run directories are immutable after training finishes;
- checkpoints, plots, metrics, and evaluation outputs belong inside the run directory;
- source code defines parameter meaning; experiment configs define run-specific values.

Run IDs use:

```text
<timestamp>_seed<seed>_<config_hash>
```

Example:

```text
20260828_171530_seed42_a83f91
```

The saved run metadata should include, when available:

```text
experiment name
seed
Git commit hash
Python version
PyTorch version
device
start time
```

## Codex Iteration Rule

Codex should modify at most about **50 lines of code per iteration**.

Each iteration must contain:

1. one narrow goal;
2. the exact files to change;
3. the expected behavior;
4. the test or manual check to run;
5. a short explanation of what was added;
6. the next intended iteration.

If a task naturally exceeds 50 lines, split it before writing code.
