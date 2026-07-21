from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import numpy as np

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
    assert window.inference_prediction_table.horizontalHeaderItem(0).text() == "序列"
    assert window.inference_preview.pixmap().isNull() is False
    assert (gui_report_dir / "experiment_curves.png").is_file()
    window.close()


def test_custom_manifest_dataset_trains_without_core_adapter_changes(tmp_path: Path) -> None:
    source_root = tmp_path / "unconventional_dataset"
    sequence_root = source_root / "sessions" / "run-z"
    rgb_root = sequence_root / "camera_left"
    depth_root = sequence_root / "range_groundtruth"
    rgb_root.mkdir(parents=True)
    depth_root.mkdir()
    with (sequence_root / "vehicle_poses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "timestamp", "x", "y", "yaw", "speed"])
        writer.writeheader()
        for index in range(8):
            frame_id = f"{index:04d}"
            np.save(rgb_root / f"rgb_{frame_id}.npy", np.full((12, 16, 3), 20 + index, dtype=np.uint8))
            np.save(depth_root / f"depth_{frame_id}.npy", np.full((12, 16), 2.0 + index * 0.1, dtype=np.float32))
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": index * 0.1,
                    "x": index * 0.2,
                    "y": 0.0,
                    "yaw": 0.0,
                    "speed": 2.0,
                }
            )
    mapped = services.save_dataset_manifest(
        dataset_id="unconventional_depth",
        dataset_root=str(source_root),
        sequences=[
            {
                "id": "run-z",
                "root": "sessions/run-z",
                "pose_csv": "vehicle_poses.csv",
                "assets": {
                    "front_rgb": "camera_left/rgb_*.npy",
                    "depth": "range_groundtruth/depth_*.npy",
                },
                "alignment": {
                    "mode": "frame_id",
                    "filename_regex": "_(?P<frame_id>\\d+)$",
                },
            }
        ],
        destination_root=tmp_path / "dataset_catalog",
    )
    mapped_root = mapped["dataset_root"]
    inspected = services.inspect_dataset(mapped_root, "manifest_dataset", "run-z")
    split = services.create_dataset_split_definition(
        mapped_root,
        "manifest_dataset",
        output_path=tmp_path / "custom_split.json",
    )
    trained = services.run_trainer_manifest_job(
        services.CONFIG_ROOT / "trainers" / "tiny_depth.yaml",
        dataset_root=mapped_root,
        output_dir=str(tmp_path / "custom_model"),
        adapter="manifest_dataset",
        sequence_id="run-z",
        split_path=split["path"],
        parameters={"epochs": 2, "max_frames": 6, "max_pixels_per_frame": 32, "max_depth_m": 20.0},
    )
    inferred = services.run_inference_manifest_job(
        services.CONFIG_ROOT / "trainers" / "tiny_depth.yaml",
        artifact_path=trained["artifact_path"],
        dataset_root=mapped_root,
        adapter="manifest_dataset",
        sequence_id="run-z",
        split_path=split["path"],
        parameters={"max_samples": 2, "split_name": "test"},
        output_dir=str(tmp_path / "custom_inference"),
    )

    assert inspected["frame_count"] == 8
    assert inspected["quality"]["available_modalities"] == ["depth", "front_rgb"]
    assert Path(trained["artifact_path"]).is_file()
    assert inferred["status"] == "completed"
    assert inferred["prediction_count"] == 1
