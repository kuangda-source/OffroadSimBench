from __future__ import annotations

import json
from pathlib import Path

from desktop_app import services
from offroad_sim.utils.yaml_io import load_yaml_file


def _write_echo_trainer(root: Path) -> Path:
    script = root / "echo_trainer.py"
    script.write_text(
        """
from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset_root")
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
checkpoint = output / "echo.ckpt"
checkpoint.write_text("checkpoint", encoding="utf-8")
print(json.dumps({
    "output_dir": str(output.resolve()),
    "checkpoint_path": str(checkpoint.resolve()),
    "artifact_type": "checkpoint",
    "metrics": {"final_loss": 0.25, "epochs": args.epochs},
    "history": {"loss": [1.0, 0.5, 0.25]},
}, indent=2))
""",
        encoding="utf-8",
    )
    manifest = root / "trainer.yaml"
    manifest.write_text(
        """
trainer_id: echo_trainer
display_name: Echo Trainer
runtime: python
entrypoint: echo_trainer.py
input:
  dataset_format: manifest_dataset
parameters:
  epochs:
    type: int
    default: 3
arguments:
  - "{dataset_root}"
  - "--output"
  - "{output_dir}"
  - "--epochs"
  - "{params.epochs}"
outputs:
  artifact_type: checkpoint
""",
        encoding="utf-8",
    )
    return manifest


def test_trainer_manifest_entries_expose_defaults(tmp_path) -> None:
    manifest = _write_echo_trainer(tmp_path)

    entries = services.trainer_manifest_entries(tmp_path)

    assert entries[0]["id"] == "echo_trainer"
    assert entries[0]["label"] == "Echo Trainer"
    assert entries[0]["manifest_path"] == str(manifest.resolve())
    assert entries[0]["parameters"]["epochs"]["default"] == 3


def test_trainer_manifest_entries_accept_named_yaml_files(tmp_path) -> None:
    manifest = _write_echo_trainer(tmp_path)
    named = tmp_path / "custom_trainer.yaml"
    manifest.replace(named)

    entries = services.trainer_manifest_entries(tmp_path)

    assert entries[0]["id"] == "echo_trainer"
    assert entries[0]["manifest_path"] == str(named.resolve())


def test_import_trainer_manifest_copies_external_manifest(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    source = _write_echo_trainer(source_dir)
    destination_root = tmp_path / "trainers"

    row = services.import_trainer_manifest(source, destination_root=destination_root)

    copied_path = Path(row["manifest_path"])
    assert copied_path.parent == destination_root
    assert copied_path.name == "echo_trainer.yaml"
    copied = load_yaml_file(copied_path)
    assert copied["entrypoint"] == str((source_dir / "echo_trainer.py").resolve())
    assert copied["imported_from"] == str(source.resolve())
    entries = services.trainer_manifest_entries(destination_root)
    assert entries[0]["id"] == "echo_trainer"


def test_save_trainer_manifest_from_entrypoint_exposes_preset(tmp_path) -> None:
    script = tmp_path / "train_custom.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    destination_root = tmp_path / "trainers"

    row = services.save_trainer_manifest(
        trainer_id="custom_path_trainer",
        label="Custom Path Trainer",
        entrypoint=str(script),
        runtime="python",
        arguments=["{dataset_root}", "--output", "{output_dir}", "--epochs", "{params.epochs}"],
        parameters={"epochs": {"type": "int", "default": 2}},
        outputs={"artifact_type": "checkpoint"},
        destination_root=destination_root,
    )

    saved = load_yaml_file(destination_root / "custom_path_trainer.yaml")
    presets = services.training_preset_entries()

    assert row["id"] == "custom_path_trainer"
    assert row["label"] == "Custom Path Trainer"
    assert saved["entrypoint"] == str(script.resolve())
    assert saved["arguments"][-1] == "{params.epochs}"
    assert any(preset["id"] == "custom_path_trainer" for preset in services.trainer_manifest_entries(destination_root))
    assert any(preset["id"] == "tiny_world_model" for preset in presets)


def test_run_trainer_manifest_job_executes_command_and_records(tmp_path) -> None:
    manifest = _write_echo_trainer(tmp_path)
    output_dir = tmp_path / "run"

    payload = services.run_trainer_manifest_job(
        str(manifest),
        dataset_root="dataset_root",
        output_dir=str(output_dir),
        parameters={"epochs": 5},
        adapter="manifest_dataset",
        sequence_id="clip_001",
    )

    record = json.loads((output_dir / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))

    assert Path(payload["checkpoint_path"]).name == "echo.ckpt"
    assert payload["training_run_path"] == str((output_dir / services.TRAINING_RUN_FILENAME).resolve())
    assert (output_dir / "stdout.log").exists()
    assert record["preset_id"] == "echo_trainer"
    assert record["dataset_root"] == "dataset_root"
    assert record["adapter"] == "manifest_dataset"
    assert record["sequence_id"] == "clip_001"
    assert record["parameters"]["epochs"] == 5
    assert record["history"]["loss"] == [1.0, 0.5, 0.25]
    assert record["logs"]["stdout"] == str((output_dir / "stdout.log").resolve())
    assert record["logs"]["stderr"] == str((output_dir / "stderr.log").resolve())


def test_run_training_config_job_executes_external_trainer_config(tmp_path) -> None:
    _write_echo_trainer(tmp_path)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_dir = tmp_path / "configured_run"
    config = {
        "id": "echo_config",
        "label": "Echo Config",
        "training_preset_id": "echo_trainer",
        "dataset_root": str(dataset_root),
        "adapter": "manifest_dataset",
        "sequence_id": "clip_001",
        "output_path": str(output_dir),
        "parameters": {"epochs": 6},
    }

    payload = services.run_training_config_job(config, trainer_root=tmp_path)

    record = json.loads((output_dir / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "checkpoint"
    assert Path(payload["artifact_path"]).name == "echo.ckpt"
    assert payload["training_config"]["id"] == "echo_config"
    assert payload["metrics"]["final_loss"] == 0.25
    assert record["parameters"]["epochs"] == 6


def test_smoke_tiny_training_config_creates_dataset_and_records_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "SMOKE_TRAINING_DATASET_ROOT", tmp_path / "smoke_orfd")
    monkeypatch.setattr(services, "SMOKE_TINY_MODEL_OUTPUT_DIR", tmp_path / "tiny_model")
    config = next(row for row in services.training_config_entries() if row["id"] == "smoke_tiny_world_model")
    config = {
        **config,
        "dataset_root": str(services.SMOKE_TRAINING_DATASET_ROOT),
        "output_path": str(services.SMOKE_TINY_MODEL_OUTPUT_DIR),
    }

    report = services.validate_training_config_setup(config, trainer_root=services.CONFIG_ROOT / "trainers")
    payload = services.run_training_config_job(config, trainer_root=services.CONFIG_ROOT / "trainers")

    record = json.loads((tmp_path / "tiny_model" / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert report["ready"] is True
    assert Path(config["dataset_root"]).exists()
    assert payload["training_config"]["id"] == "smoke_tiny_world_model"
    assert payload["artifact_type"] == "world_model"
    assert record["preset_id"] == "tiny_world_model_script"
    assert "train_rmse" in record["metrics"]
    assert record["history"]["train_rmse"]


def test_validate_training_config_setup_previews_external_trainer_command(tmp_path) -> None:
    manifest = _write_echo_trainer(tmp_path)
    config = services.save_training_config(
        config_id="echo_config",
        label="Echo Config",
        training_preset_id="echo_trainer",
        dataset_root=str(tmp_path / "dataset"),
        adapter="manifest_dataset",
        sequence_id="clip_001",
        output_path=str(tmp_path / "run"),
        parameters={"epochs": "5"},
        path=tmp_path / "training_configs.json",
    )
    (tmp_path / "dataset").mkdir()

    report = services.validate_training_config_setup(config, trainer_root=tmp_path)

    assert report["ready"] is True
    assert report["status"] == "ready"
    assert report["issues"] == []
    assert report["training_preset"]["id"] == "echo_trainer"
    assert report["parameters"] == {"epochs": 5}
    assert report["dataset"]["exists"] is True
    assert report["command_preview"][1] == str((tmp_path / "echo_trainer.py").resolve())
    assert "--epochs" in report["command_preview"]
    assert "5" in report["command_preview"]
    assert report["training_preset"]["manifest_path"] == str(manifest.resolve())


def test_validate_training_config_setup_reports_missing_required_parameter(tmp_path) -> None:
    script = tmp_path / "train_required.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
trainer_id: required_trainer
display_name: Required Trainer
runtime: python
entrypoint: train_required.py
parameters:
  learning_rate:
    type: float
    required: true
arguments:
  - "{dataset_root}"
  - "--output"
  - "{output_dir}"
  - "--learning-rate"
  - "{params.learning_rate}"
outputs:
  artifact_type: checkpoint
""",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    report = services.validate_training_config_setup(
        {
            "id": "required_config",
            "label": "Required Config",
            "training_preset_id": "required_trainer",
            "dataset_root": str(dataset_root),
            "adapter": "manifest_dataset",
            "sequence_id": "",
            "output_path": str(tmp_path / "run"),
            "parameters": {},
        },
        trainer_root=tmp_path,
    )

    assert report["ready"] is False
    assert report["status"] == "invalid"
    assert "Missing required parameter: learning_rate" in report["issues"]


def test_save_script_training_config_infers_schema_and_argument_template(tmp_path) -> None:
    script = tmp_path / "train_any.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    bundle = services.save_script_training_config(
        label="Any Driving Model",
        trainer_entrypoint=str(script),
        dataset_root=str(dataset_root),
        adapter="manifest_dataset",
        sequence_id="clip_001",
        output_path=str(tmp_path / "run"),
        parameters={"epochs": 4, "learning_rate": 0.001, "use_amp": True, "model_name": "tiny"},
        trainer_destination_root=tmp_path / "trainers",
        training_config_path=tmp_path / "training_configs.json",
    )

    trainer = bundle["trainer"]
    config = bundle["training_config"]
    manifest = load_yaml_file(Path(trainer["manifest_path"]))

    assert trainer["id"] == "any_driving_model"
    assert config["training_preset_id"] == "any_driving_model"
    assert config["dataset_root"] == str(dataset_root)
    assert config["adapter"] == "manifest_dataset"
    assert config["sequence_id"] == "clip_001"
    assert config["parameters"]["epochs"] == 4
    assert manifest["parameters"]["epochs"] == {"type": "int", "default": 4}
    assert manifest["parameters"]["learning_rate"] == {"type": "float", "default": 0.001}
    assert manifest["parameters"]["use_amp"] == {"type": "bool", "default": True}
    assert manifest["parameters"]["model_name"] == {"type": "str", "default": "tiny"}
    assert manifest["arguments"][:3] == ["{dataset_root}", "--output", "{output_dir}"]
    assert "--epochs" in manifest["arguments"]
    assert "{params.epochs}" in manifest["arguments"]


def test_script_training_config_executes_and_records_metrics(tmp_path) -> None:
    script = tmp_path / "train_any.py"
    script.write_text(
        """
from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset_root")
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--use-amp", required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
checkpoint = output / "model.ckpt"
checkpoint.write_text("checkpoint", encoding="utf-8")
print(json.dumps({
    "checkpoint_path": str(checkpoint),
    "artifact_type": "checkpoint",
    "metrics": {"final_loss": 0.2, "epochs": args.epochs},
    "history": {"loss": [0.7, 0.4, 0.2]},
}, indent=2))
""",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_dir = tmp_path / "run"
    bundle = services.save_script_training_config(
        label="Script Runner",
        trainer_entrypoint=str(script),
        dataset_root=str(dataset_root),
        output_path=str(output_dir),
        parameters={"epochs": 3, "learning_rate": 0.01, "use_amp": False},
        trainer_destination_root=tmp_path / "trainers",
        training_config_path=tmp_path / "training_configs.json",
    )

    payload = services.run_training_config_job(bundle["training_config"], trainer_root=tmp_path / "trainers")

    record = json.loads((output_dir / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "checkpoint"
    assert Path(payload["artifact_path"]).name == "model.ckpt"
    assert payload["metrics"]["final_loss"] == 0.2
    assert record["history"]["loss"] == [0.7, 0.4, 0.2]
    assert record["parameters"]["learning_rate"] == 0.01
