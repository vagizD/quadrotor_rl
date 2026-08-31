import json
from datetime import datetime, timezone

from experiments import (
    create_run_directory,
    load_experiment_config,
)


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def test_run_directory_freezes_config_and_metadata(tmp_path) -> None:
    config = load_experiment_config(CONFIG_PATH)
    start_time = datetime(2026, 8, 29, 12, 34, 56, tzinfo=timezone.utc)

    run = create_run_directory(
        config,
        runs_root=tmp_path,
        device="cpu",
        start_time=start_time,
        git_commit="test-commit",
    )

    assert run.run_id.startswith("20260829_123456_seed42_")
    assert len(run.run_id.rsplit("_", maxsplit=1)[1]) == 8
    assert run.run_dir == tmp_path / config.name / run.run_id
    assert run.config_path.exists()
    assert run.metadata_path.exists()
    assert run.checkpoints_dir.is_dir()
    assert run.plots_dir.is_dir()
    assert run.evaluation_dir.is_dir()
    assert run.tensorboard_dir.is_dir()
    assert load_experiment_config(run.config_path).as_dict() == config.as_dict()

    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert metadata["experiment_name"] == config.name
    assert metadata["seed"] == config.seed
    assert metadata["git_commit"] == "test-commit"
    assert metadata["device"] == "cpu"
    assert metadata["start_time"] == start_time.isoformat()
