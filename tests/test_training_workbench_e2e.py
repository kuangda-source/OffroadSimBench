from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from desktop_app import services
from desktop_app.qt_main import MainWindow
from offroad_sim.datasets import create_mock_orfd_dataset
from offroad_sim.training import TrainingJobQueue


def test_training_workbench_dataset_to_report_closed_loop(tmp_path: Path, monkeypatch) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=9)
    sequence_id = "training/seq_0001"
    manifest = services.CONFIG_ROOT / "trainers" / "tiny_depth.yaml"

    inspected = services.inspect_dataset(str(dataset_root), "orfd", sequence_id)
    quality = services.analyze_dataset_quality(
        str(dataset_root),
        "orfd",
        [sequence_id],
        output_dir=tmp_path / "quality",
    )
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        output_path=tmp_path / "split.json",
    )
    assert inspected["frame_count"] == 9
    assert quality["training_ready"] is True
    assert Path(split["path"]).is_file()

    base_parameters = {
        "epochs": 4,
        "learning_rate": 0.05,
        "ridge": 0.0001,
        "max_frames": 6,
        "max_pixels_per_frame": 64,
        "max_depth_m": 20.0,
        "seed": 7,
    }
    report = services.validate_training_config_setup(
        {
            "id": "tiny_depth_acceptance",
            "label": "Tiny depth acceptance",
            "training_preset_id": "tiny_rgb_depth",
            "dataset_root": str(dataset_root),
            "adapter": "orfd",
            "sequence_id": sequence_id,
            "split_path": split["path"],
            "output_path": str(tmp_path / "run_a"),
            "parameters": base_parameters,
        }
    )
    assert report["ready"] is True

    queue = TrainingJobQueue(max_parallel=1)
    try:
        first = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(dataset_root),
            output_dir=str(tmp_path / "run_a"),
            adapter="orfd",
            sequence_id=sequence_id,
            split_path=split["path"],
            parameters=base_parameters,
        )
        second = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(dataset_root),
            output_dir=str(tmp_path / "run_b"),
            adapter="orfd",
            sequence_id=sequence_id,
            split_path=split["path"],
            parameters={**base_parameters, "ridge": 0.02},
        )
        assert first.wait(timeout=20.0)["status"] == "completed"
        assert second.wait(timeout=20.0)["status"] == "completed"
    finally:
        queue.close(cancel_running=True)

    runs = services.training_run_entries(tmp_path)
    artifacts = services.training_artifact_entries(tmp_path)
    assert len(runs) == 2
    assert len(artifacts) == 2
    assert all(row["inference_available"] for row in artifacts)
    assert all(row["parameters"]["epochs"] == 4 for row in artifacts)

    inference = services.run_inference_manifest_job(
        manifest,
        artifact_path=artifacts[0]["artifact_path"],
        dataset_root=str(dataset_root),
        adapter="orfd",
        sequence_id=sequence_id,
        split_path=split["path"],
        parameters={"max_samples": 3, "split_name": "test"},
        output_dir=str(tmp_path / "inference"),
    )
    comparison = services.compare_training_runs(runs, metric="validation_loss")
    exported = services.export_experiment_report(comparison, tmp_path / "report")
    assert inference["metrics"]["depth_rmse_m"] > 0.0
    assert inference["metrics"]["split_name"] == "test"
    assert inference["prediction_count"] == split["counts"]["test"]
    assert Path(inference["previews"]["primary"]).is_file()
    assert comparison["run_count"] == 2
    assert Path(exported["markdown_path"]).is_file()
    assert Path(exported["html_path"]).is_file()

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.catalog["training_runs"] = runs
    window.catalog["training_artifacts"] = artifacts
    window._fill_training_run_list()
    window._fill_training_artifact_combo()
    assert json.loads(window.inference_params_edit.toPlainText())["split_name"] == "test"
    window.training_run_list.item(0).setSelected(True)
    window.training_run_list.item(1).setSelected(True)
    window.compare_selected_training_runs()
    window._inference_finished(inference)
    gui_report_dir = tmp_path / "gui_report"
    monkeypatch.setattr(
        "desktop_app.qt_main.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(gui_report_dir),
    )
    window.export_experiment_comparison_report()

    assert window.experiment_comparison_table.rowCount() == 2
    assert window.inference_prediction_table.horizontalHeaderItem(0).text() == "frame_id"
    assert window.inference_preview.pixmap().isNull() is False
    assert (gui_report_dir / "experiment_curves.png").is_file()
    window.close()
