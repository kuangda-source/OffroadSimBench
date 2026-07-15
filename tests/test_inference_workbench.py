from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from desktop_app import services
from desktop_app.qt_main import MainWindow
from offroad_sim.datasets import create_mock_orfd_dataset


def test_tiny_manifest_train_checkpoint_infer_and_preview(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=8)
    manifest = services.CONFIG_ROOT / "trainers" / "tiny_world_model.yaml"
    model_dir = tmp_path / "model"

    trained = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(dataset_root),
        output_dir=str(model_dir),
        parameters={"ridge": 0.0001},
        adapter="orfd",
        sequence_id="training/seq_0001",
    )
    report = services.validate_inference_setup(
        manifest,
        artifact_path=trained["artifact_path"],
        dataset_root=str(dataset_root),
        adapter="orfd",
        sequence_id="training/seq_0001",
        parameters={"max_samples": 4},
        output_dir=str(tmp_path / "inference"),
    )
    inferred = services.run_inference_manifest_job(
        manifest,
        artifact_path=trained["artifact_path"],
        dataset_root=str(dataset_root),
        adapter="orfd",
        sequence_id="training/seq_0001",
        parameters={"max_samples": 4},
        output_dir=str(tmp_path / "inference"),
    )

    assert report["ready"] is True
    assert report["command_preview"]
    assert trained["artifact_path"].endswith("model.json")
    assert inferred["status"] == "completed"
    assert inferred["prediction_count"] == 4
    assert inferred["metrics"]["sample_count"] == 4
    assert "position_rmse" in inferred["metrics"]
    assert Path(inferred["previews"]["trajectory"]).is_file()
    assert Path(inferred["path"]).name == services.INFERENCE_RUN_FILENAME
    predictions = json.loads((tmp_path / "inference" / "predictions.json").read_text(encoding="utf-8"))
    assert len(predictions) == 4


def test_training_artifact_catalog_reports_inference_capability(tmp_path) -> None:
    artifact = tmp_path / "run" / "model.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"model_type": "tiny_learned"}', encoding="utf-8")
    manifest = services.CONFIG_ROOT / "trainers" / "tiny_world_model.yaml"
    services.write_training_run_record(
        artifact.parent,
        preset_id="tiny_world_model_script",
        status="completed",
        artifact_path=str(artifact),
        artifact_type="world_model",
        metrics={"validation_rmse": 0.2},
        summary={"trainer_manifest_path": str(manifest.resolve())},
    )

    entries = services.training_artifact_entries(tmp_path)

    assert entries[0]["artifact_path"] == str(artifact.resolve())
    assert entries[0]["exists"] is True
    assert entries[0]["size_bytes"] > 0
    assert entries[0]["inference_available"] is True
    assert entries[0]["metrics"]["validation_rmse"] == 0.2


def test_inference_validation_rejects_missing_capability_and_artifact(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        "schema_version: 1\ntrainer_id: train_only\nlaunch:\n  kind: python_script\n  entrypoint: train.py\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    report = services.validate_inference_setup(
        manifest,
        artifact_path=str(tmp_path / "missing.ckpt"),
        dataset_root=str(dataset),
    )

    assert report["ready"] is False
    issues = " ".join(report["issues"])
    assert "does not declare an inference" in issues
    assert "artifact not found" in issues.lower()


def test_import_rebases_bare_python_training_and_inference_scripts(tmp_path) -> None:
    source = tmp_path / "external"
    source.mkdir()
    (source / "train.py").write_text("print('{}')\n", encoding="utf-8")
    (source / "infer.py").write_text("print('{}')\n", encoding="utf-8")
    manifest = source / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: bare_scripts
launch:
  kind: python_script
  entrypoint: train.py
inference:
  launch:
    kind: python_script
    entrypoint: infer.py
""",
        encoding="utf-8",
    )

    imported = services.import_trainer_manifest(manifest, destination_root=tmp_path / "catalog")

    assert imported["launch"]["entrypoint"] == str((source / "train.py").resolve())
    assert imported["inference"]["launch"]["entrypoint"] == str((source / "infer.py").resolve())


def test_missing_inference_script_is_not_reported_available(tmp_path) -> None:
    train_script = tmp_path / "train.py"
    train_script.write_text("print('{}')\n", encoding="utf-8")
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: missing_infer
launch:
  kind: python_script
  entrypoint: train.py
inference:
  launch:
    kind: python_script
    entrypoint: missing_infer.py
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "run" / "model.ckpt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model")
    services.write_training_run_record(
        artifact.parent,
        preset_id="missing_infer",
        status="completed",
        artifact_path=str(artifact),
        artifact_type="checkpoint",
        summary={"trainer_manifest_path": str(manifest)},
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    report = services.validate_inference_setup(
        manifest,
        artifact_path=str(artifact),
        dataset_root=str(dataset),
    )
    entries = services.training_artifact_entries(tmp_path)

    assert report["ready"] is False
    assert "Inference launch is invalid" in " ".join(report["issues"])
    assert entries[0]["inference_available"] is False


def test_gui_runs_selected_checkpoint_inference_and_renders_results(monkeypatch, tmp_path) -> None:
    QApplication.instance() or QApplication([])
    dataset_root = create_mock_orfd_dataset(tmp_path / "gui_dataset", frame_count=7)
    manifest = services.CONFIG_ROOT / "trainers" / "tiny_world_model.yaml"
    trained = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(dataset_root),
        output_dir=str(tmp_path / "gui_model"),
        adapter="orfd",
        sequence_id="training/seq_0001",
    )
    artifact = services.training_artifact_entries(tmp_path)[0]
    window = MainWindow()
    window.catalog["training_artifacts"] = [artifact]
    window._fill_training_artifact_combo()
    window.dataset_root_edit.setText(str(dataset_root))
    window.adapter_edit.setText("orfd")
    window.sequence_combo.setCurrentText("training/seq_0001")
    monkeypatch.setattr(window, "_run_task", lambda fn, callback, failure_label, **kwargs: callback(fn()))

    window.run_selected_artifact_inference()

    assert json.loads(window.model_summary.toPlainText())["artifact_path"] == trained["artifact_path"]
    assert "position_rmse" in window.inference_metric_summary.text()
    assert window.inference_prediction_table.rowCount() == 6
    assert window.inference_preview.pixmap().isNull() is False
    assert "inference_run.json" in window.model_summary.toPlainText()
    window.close()
