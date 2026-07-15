from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QSpinBox

from desktop_app import services
from desktop_app.qt_main import MainWindow
from offroad_sim.datasets import create_mock_orfd_dataset
from offroad_sim.training import build_trainer_command, normalize_trainer_manifest, validate_trainer_parameters


def _ensure_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _write_result_trainer(path: Path, body: str) -> None:
    path.write_text(
        "from __future__ import annotations\n"
        "import json, os\n"
        "from pathlib import Path\n"
        + body
        + "\nprint(json.dumps({'artifact_path': str(Path('artifact.bin').resolve()), "
        "'artifact_type': 'checkpoint', 'metrics': {'loss': 0.1}, "
        "'history': {'loss': [0.3, 0.1]}, 'cwd': str(Path.cwd()), "
        "'marker': os.environ.get('TRAINER_MARKER', '')}))\n",
        encoding="utf-8",
    )


def test_parameter_schema_enforces_types_ranges_enum_and_dependency(tmp_path) -> None:
    manifest_path = tmp_path / "trainer.yaml"
    normalized = normalize_trainer_manifest(
        {
            "schema_version": 1,
            "trainer_id": "schema_test",
            "launch": {"kind": "python_module", "module": "trainer"},
            "parameters": {
                "epochs": {"type": "int", "default": 3, "min": 1, "max": 10},
                "optimizer": {"type": "str", "default": "adam", "enum": ["adam", "sgd"]},
                "use_amp": {"type": "bool", "default": False},
                "checkpoint": {"type": "file", "default": "model.ckpt"},
                "output_dir": {"type": "directory", "default": "outputs/run"},
                "amp_dtype": {
                    "type": "str",
                    "required": True,
                    "enum": ["float16", "bfloat16"],
                    "depends_on": {"parameter": "use_amp", "equals": True},
                },
            },
        },
        manifest_path=manifest_path,
    )

    values = validate_trainer_parameters(normalized["parameters"], {"epochs": "5", "use_amp": False})

    assert values == {
        "epochs": 5,
        "optimizer": "adam",
        "use_amp": False,
        "checkpoint": "model.ckpt",
        "output_dir": "outputs/run",
    }
    with pytest.raises(ValueError, match="amp_dtype"):
        validate_trainer_parameters(normalized["parameters"], {"use_amp": True})
    with pytest.raises(ValueError, match=">= 1"):
        validate_trainer_parameters(normalized["parameters"], {"epochs": 0, "use_amp": False})
    with pytest.raises(ValueError, match="must be one of"):
        validate_trainer_parameters(normalized["parameters"], {"optimizer": "bad", "use_amp": False})
    with pytest.raises(ValueError, match="Unknown trainer parameter"):
        validate_trainer_parameters(normalized["parameters"], {"extra": 1})


def test_python_module_launch_uses_working_directory_and_environment(tmp_path) -> None:
    module = tmp_path / "module_trainer.py"
    _write_result_trainer(module, "Path('artifact.bin').write_text('ok', encoding='utf-8')")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: module_trainer
launch:
  kind: python_module
  module: module_trainer
  working_directory: .
  environment:
    PYTHONPATH: "{manifest_dir}"
    TRAINER_MARKER: module-ok
outputs:
  artifact_type: checkpoint
""",
        encoding="utf-8",
    )

    payload = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "run"),
    )

    assert payload["marker"] == "module-ok"
    assert payload["cwd"] == str(tmp_path.resolve())
    assert Path(payload["artifact_path"]).name == "artifact.bin"


def test_executable_launch_runs_without_platform_code_changes(tmp_path) -> None:
    script = tmp_path / "executable_target.py"
    _write_result_trainer(script, "Path('artifact.bin').write_text('ok', encoding='utf-8')")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "trainer_id: executable_trainer",
                "launch:",
                "  kind: executable",
                f"  entrypoint: '{sys.executable}'",
                "  working_directory: .",
                "arguments:",
                f"  - '{script}'",
                "outputs:",
                "  artifact_type: checkpoint",
            ]
        ),
        encoding="utf-8",
    )

    payload = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "run"),
    )

    assert payload["metrics"]["loss"] == 0.1
    assert payload["history"]["loss"] == [0.3, 0.1]


def test_conda_prefix_python_script_resolves_environment_python(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    manifest = normalize_trainer_manifest(
        {
            "schema_version": 1,
            "trainer_id": "conda_prefix",
            "launch": {
                "kind": "python_script",
                "entrypoint": str(script),
                "conda_env": str(Path(sys.executable).parent),
            },
        },
        manifest_path=tmp_path / "trainer.yaml",
    )

    command = build_trainer_command(manifest, arguments=[], manifest_dir=tmp_path)

    assert Path(command[0]).resolve() == Path(sys.executable).resolve()
    assert Path(command[1]).resolve() == script.resolve()


def test_named_conda_environment_uses_environment_python(monkeypatch, tmp_path) -> None:
    conda = tmp_path / "conda.exe"
    conda.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: str(conda) if name == "conda" else None)
    manifest = normalize_trainer_manifest(
        {
            "schema_version": 1,
            "trainer_id": "named_conda",
            "launch": {
                "kind": "python_module",
                "module": "package.train",
                "conda_env": "model-env",
            },
        },
        manifest_path=tmp_path / "trainer.yaml",
    )

    command = build_trainer_command(manifest, arguments=[], manifest_dir=tmp_path)

    assert command == [str(conda), "run", "-n", "model-env", "python", "-m", "package.train"]


def test_training_validation_checks_adapter_modalities_and_split(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=3)
    script = tmp_path / "train.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: modality_trainer
launch:
  kind: python_script
  entrypoint: train.py
input:
  dataset_format: orfd
  required_modalities: [front_rgb, lidar_points]
  split_required: true
""",
        encoding="utf-8",
    )
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        output_path=tmp_path / "split.json",
    )
    config = {
        "id": "compatible",
        "training_preset_id": "modality_trainer",
        "dataset_root": str(dataset_root),
        "adapter": "orfd",
        "sequence_id": "training/seq_0001",
        "split_path": split["path"],
        "output_path": str(tmp_path / "run"),
        "parameters": {},
    }

    report = services.validate_training_config_setup(config, trainer_root=tmp_path)

    assert report["ready"] is True
    assert report["compatibility"]["compatible"] is True
    assert report["compatibility"]["required_modalities"] == ["front_rgb", "lidar_points"]
    assert {"front_rgb", "lidar_points"}.issubset(report["compatibility"]["available_modalities"])

    missing_split = services.validate_training_config_setup(
        {**config, "split_path": ""},
        trainer_root=tmp_path,
    )
    assert missing_split["ready"] is False
    assert "requires a dataset split" in " ".join(missing_split["issues"])

    other_dataset_root = create_mock_orfd_dataset(tmp_path / "other_dataset", frame_count=2)
    wrong_split = services.create_dataset_split_definition(
        str(other_dataset_root),
        "orfd",
        output_path=tmp_path / "wrong_split.json",
    )
    mismatched = services.validate_training_config_setup(
        {**config, "split_path": wrong_split["path"]},
        trainer_root=tmp_path,
    )
    assert mismatched["ready"] is False
    assert "split root" in " ".join(mismatched["issues"])


def test_training_validation_rejects_duplicate_and_overlapping_split_samples(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=6)
    script = tmp_path / "train.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: split_guard
launch:
  kind: python_script
  entrypoint: train.py
input:
  dataset_format: orfd
  split_required: true
""",
        encoding="utf-8",
    )
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        output_path=tmp_path / "split.json",
    )
    payload = json.loads(Path(split["path"]).read_text(encoding="utf-8"))
    payload["splits"]["validation"][0]["frame_indices"] = [0, 0]
    payload["splits"]["test"][0]["frame_indices"] = [0]
    Path(split["path"]).write_text(json.dumps(payload), encoding="utf-8")

    report = services.validate_training_config_setup(
        {
            "id": "leaky",
            "training_preset_id": "split_guard",
            "dataset_root": str(dataset_root),
            "adapter": "orfd",
            "split_path": split["path"],
            "parameters": {},
        },
        trainer_root=tmp_path,
    )

    assert report["ready"] is False
    assert "split" in " ".join(report["issues"]).lower()


def test_declared_missing_artifact_fails_sync_and_async_runs(tmp_path) -> None:
    script = tmp_path / "missing.py"
    script.write_text(
        "import json\nprint(json.dumps({'artifact_path': 'missing.ckpt'}))\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: missing_artifact
launch:
  kind: python_script
  entrypoint: missing.py
""",
        encoding="utf-8",
    )

    sync_dir = tmp_path / "sync"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        services.run_trainer_manifest_job(manifest, dataset_root=str(tmp_path), output_dir=str(sync_dir))
    assert json.loads((sync_dir / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))["status"] == "failed"

    from offroad_sim.training import TrainingJobQueue

    queue = TrainingJobQueue(max_parallel=1)
    try:
        job = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "async"),
        )
        finished = job.wait(timeout=10.0)
    finally:
        queue.close(cancel_running=True)
    assert finished["status"] == "failed"
    assert "does not exist" in finished["error"]
    assert json.loads((tmp_path / "async" / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))["status"] == "failed"


def test_historical_rerun_rejects_modified_trainer_entrypoint(tmp_path) -> None:
    script = tmp_path / "trainer.py"
    _write_result_trainer(script, "Path('artifact.bin').write_text('ok', encoding='utf-8')")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: code_hash
launch:
  kind: python_script
  entrypoint: trainer.py
  working_directory: .
""",
        encoding="utf-8",
    )
    first = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "first"),
    )
    record = json.loads(Path(first["training_run_path"]).read_text(encoding="utf-8"))
    script.write_text(script.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="entrypoint failed SHA-256 verification"):
        services.run_trainer_manifest_job(
            record["trainer_manifest_snapshot_path"],
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "rerun"),
        )


def test_legacy_top_level_entrypoint_snapshot_is_portable_and_hashed(tmp_path) -> None:
    script = tmp_path / "legacy.py"
    _write_result_trainer(script, "Path('artifact.bin').write_text('ok', encoding='utf-8')")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: legacy_entrypoint
entrypoint: legacy.py
""",
        encoding="utf-8",
    )
    first = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "first"),
    )
    record = json.loads(Path(first["training_run_path"]).read_text(encoding="utf-8"))

    rerun = services.run_trainer_manifest_job(
        record["trainer_manifest_snapshot_path"],
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "rerun"),
    )

    assert Path(record["trainer_entrypoint_path"]) == script.resolve()
    assert len(record["trainer_entrypoint_sha256"]) == 64
    assert Path(rerun["artifact_path"]).is_file()


def test_gui_builds_typed_parameter_controls_and_dependencies() -> None:
    _ensure_app()
    window = MainWindow()
    window.catalog["training_presets"] = [
        {
            "id": "typed_trainer",
            "label": "Typed Trainer",
            "available": True,
            "parameters": {
                "epochs": {"type": "int", "default": 4, "min": 1, "max": 20},
                "learning_rate": {"type": "float", "default": 0.01, "min": 0.0001, "max": 1.0},
                "optimizer": {"type": "str", "default": "adam", "enum": ["adam", "sgd"]},
                "use_amp": {"type": "bool", "default": False},
                "checkpoint": {"type": "file", "default": "model.ckpt"},
                "output_dir": {"type": "directory", "default": "outputs/run"},
                "amp_dtype": {
                    "type": "str",
                    "default": "float16",
                    "enum": ["float16", "bfloat16"],
                    "depends_on": {"parameter": "use_amp", "equals": True},
                },
            },
        }
    ]

    window._fill_training_preset_combo()

    assert isinstance(window.trainer_parameter_controls["epochs"], QSpinBox)
    assert isinstance(window.trainer_parameter_controls["learning_rate"], QDoubleSpinBox)
    assert isinstance(window.trainer_parameter_controls["optimizer"], QComboBox)
    assert isinstance(window.trainer_parameter_controls["use_amp"], QCheckBox)
    assert isinstance(window.trainer_parameter_controls["checkpoint"], QLineEdit)
    assert isinstance(window.trainer_parameter_controls["output_dir"], QLineEdit)
    browse_buttons = [
        button for button in window.trainer_parameter_form.findChildren(QPushButton) if button.text() == "选择"
    ]
    assert len(browse_buttons) == 2
    assert window.trainer_parameter_controls["amp_dtype"].isEnabled() is False
    window.trainer_parameter_controls["use_amp"].setChecked(True)
    assert window.trainer_parameter_controls["amp_dtype"].isEnabled() is True
    window.trainer_parameter_controls["epochs"].setValue(9)
    assert window._trainer_parameters_from_text()["epochs"] == 9
    assert json.loads(window.trainer_params_edit.toPlainText())["epochs"] == 9
    assert window._trainer_parameters_from_text()["checkpoint"] == "model.ckpt"
    window.close()
