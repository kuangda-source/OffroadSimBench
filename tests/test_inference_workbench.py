from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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


def test_standalone_inference_manifest_is_loaded_and_embedded_on_import(tmp_path) -> None:
    source = tmp_path / "external"
    source.mkdir()
    (source / "train.py").write_text("print('{}')\n", encoding="utf-8")
    (source / "infer.py").write_text("print('{}')\n", encoding="utf-8")
    trainer = source / "trainer.yaml"
    trainer.write_text(
        """
schema_version: 1
trainer_id: external_inference
launch:
  kind: python_script
  entrypoint: train.py
inference_manifest: inference.yaml
""",
        encoding="utf-8",
    )
    inference = source / "inference.yaml"
    inference.write_text(
        """
schema_version: 1
inference_id: external_inference
launch:
  kind: python_script
  entrypoint: infer.py
parameters:
  max_samples:
    type: int
    default: 4
arguments:
  - "{artifact_path}"
  - "{dataset_root}"
""",
        encoding="utf-8",
    )

    loaded = services.load_trainer_manifest(trainer)
    discovered = services.trainer_manifest_entries(source)
    imported = services.import_trainer_manifest(trainer, destination_root=tmp_path / "catalog")
    installed_yaml = services.load_yaml_file(Path(imported["manifest_path"]))

    assert loaded["inference_manifest_path"] == str(inference.resolve())
    assert [row["id"] for row in discovered] == ["external_inference"]
    assert loaded["inference"]["launch"]["entrypoint"] == str((source / "infer.py").resolve())
    assert services.inference_parameter_defaults(trainer) == {"max_samples": 4}
    assert "inference_manifest" not in installed_yaml
    assert installed_yaml["inference"]["launch"]["entrypoint"] == str((source / "infer.py").resolve())


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


def test_artifact_catalog_marks_latest_best_favorite_and_metadata(tmp_path) -> None:
    first_artifact = tmp_path / "first" / "model.ckpt"
    first_artifact.parent.mkdir()
    first_artifact.write_text("first", encoding="utf-8")
    first = services.write_training_run_record(
        first_artifact.parent,
        preset_id="tiny_world_model_script",
        status="completed",
        artifact_path=str(first_artifact),
        artifact_type="checkpoint",
        metrics={"validation_loss": 0.3, "epoch": 4},
        parameters={"learning_rate": 0.01},
    )
    second_artifact = tmp_path / "second" / "model.ckpt"
    second_artifact.parent.mkdir()
    second_artifact.write_text("second", encoding="utf-8")
    second = services.write_training_run_record(
        second_artifact.parent,
        preset_id="tiny_world_model_script",
        status="completed",
        artifact_path=str(second_artifact),
        artifact_type="checkpoint",
        metrics={"validation_loss": 0.2, "epoch": 5},
        parameters={"learning_rate": 0.001},
    )
    services.set_training_artifact_favorite(second["path"], True)
    services.mark_best_training_run(second["path"], metric="validation_loss", direction="min", value=0.2)

    entries = services.training_artifact_entries(tmp_path)

    latest = next(row for row in entries if row["artifact_path"] == str(second_artifact.resolve()))
    older = next(row for row in entries if row["artifact_path"] == str(first_artifact.resolve()))
    assert latest["latest"] is True
    assert latest["best"] is True
    assert latest["favorite"] is True
    assert latest["epoch"] == 5.0
    assert latest["parameters"]["learning_rate"] == 0.001
    assert older["latest"] is False


def test_favoriting_an_old_artifact_does_not_make_it_latest(tmp_path) -> None:
    records = []
    for name, created_at in (("older", "2026-01-01T00:00:00+0800"), ("newer", "2026-01-02T00:00:00+0800")):
        artifact = tmp_path / name / "model.ckpt"
        artifact.parent.mkdir()
        artifact.write_text(name, encoding="utf-8")
        record = services.write_training_run_record(
            artifact.parent,
            preset_id="tiny_world_model_script",
            status="completed",
            artifact_path=str(artifact),
            artifact_type="checkpoint",
        )
        payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        payload["created_at"] = created_at
        Path(record["path"]).write_text(json.dumps(payload), encoding="utf-8")
        records.append(record)

    services.set_training_artifact_favorite(records[0]["path"], True)
    entries = services.training_artifact_entries(tmp_path)

    latest = next(row for row in entries if row["latest"])
    assert Path(latest["artifact_path"]).parent.name == "newer"


def test_delete_training_artifact_checks_config_references_and_preserves_run(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    monkeypatch.setattr(services, "CONFIG_ROOT", config_root)
    artifact = tmp_path / "outputs" / "run" / "model.ckpt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("checkpoint", encoding="utf-8")
    record = services.write_training_run_record(
        artifact.parent,
        preset_id="tiny_world_model_script",
        status="completed",
        artifact_path=str(artifact),
        artifact_type="checkpoint",
    )
    reference = config_root / "world_model_configs.json"
    reference.write_text(json.dumps({"model_path": str(artifact.resolve())}), encoding="utf-8")

    with pytest.raises(ValueError, match="referenced by configuration"):
        services.delete_training_artifact(record["path"], output_root=tmp_path / "outputs")

    reference.unlink()
    payload = services.delete_training_artifact(record["path"], output_root=tmp_path / "outputs")

    assert payload["deleted"] is True
    assert not artifact.exists()
    preserved = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    assert preserved["artifact_deleted_at"]


def test_relative_config_reference_protects_artifact_from_delete(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    monkeypatch.setattr(services, "CONFIG_ROOT", config_root)
    artifact = tmp_path / "outputs" / "run" / "model.ckpt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("checkpoint", encoding="utf-8")
    record = services.write_training_run_record(
        artifact.parent,
        preset_id="relative_reference",
        status="completed",
        artifact_path=str(artifact),
        artifact_type="checkpoint",
    )
    relative = artifact.relative_to(tmp_path).as_posix()
    (config_root / "model.yaml").write_text(f"model_path: ../{relative}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="referenced by configuration"):
        services.delete_training_artifact(record["path"], output_root=tmp_path / "outputs")
    assert artifact.is_file()


def test_inference_postprocessing_failure_persists_failed_record_and_parameters(tmp_path) -> None:
    artifact = tmp_path / "model.ckpt"
    artifact.write_text("checkpoint", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    script = tmp_path / "infer.py"
    script.write_text(
        "from pathlib import Path\n"
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--limit'); a=p.parse_args()\n"
        "o=Path(a.output); o.mkdir(parents=True, exist_ok=True); (o/'predictions.json').write_text('{broken', encoding='utf-8')\n"
        "print('{}')\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: broken_inference
launch:
  kind: python_script
  entrypoint: infer.py
inference:
  launch:
    kind: python_script
    entrypoint: infer.py
  parameters:
    limit:
      type: int
      default: 3
  arguments: [--output, "{output_dir}", --limit, "{params.limit}"]
  outputs:
    predictions_file: predictions.json
""",
        encoding="utf-8",
    )
    output = tmp_path / "inference"

    with pytest.raises(RuntimeError, match="post-processing failed"):
        services.run_inference_manifest_job(
            manifest,
            artifact_path=str(artifact),
            dataset_root=str(dataset),
            parameters={"limit": 7},
            output_dir=str(output),
        )

    record = json.loads((output / services.INFERENCE_RUN_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["parameters"] == {"limit": 7}
    assert record["logs"]["stdout"].endswith("stdout.log")
