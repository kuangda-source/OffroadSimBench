from __future__ import annotations

import json
from pathlib import Path

import pytest

from desktop_app import services
from offroad_sim.datasets import create_mock_orfd_dataset


def test_tiny_depth_manifest_trains_and_runs_inference(tmp_path: Path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=8)
    manifest = services.CONFIG_ROOT / "trainers" / "tiny_depth.yaml"
    model_dir = tmp_path / "model"
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        output_path=tmp_path / "split.json",
    )

    training = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(dataset_root),
        output_dir=str(model_dir),
        adapter="orfd",
        sequence_id="training/seq_0001",
        split_path=split["path"],
        parameters={
            "epochs": 5,
            "learning_rate": 0.05,
            "ridge": 0.0001,
            "max_frames": 6,
            "max_pixels_per_frame": 64,
            "max_depth_m": 20.0,
            "seed": 7,
        },
    )
    record = json.loads(Path(training["training_run_path"]).read_text(encoding="utf-8"))
    defaults = services.inference_parameter_defaults(manifest)

    assert Path(training["artifact_path"]).name == "model.json"
    assert record["artifact_type"] == "depth_model"
    assert len(record["history"]["train_loss"]) == 5
    assert len(record["history"]["validation_loss"]) == 5
    assert record["history_steps"]["validation_loss"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert record["metrics"]["validation_rmse_m"] > 0.0
    assert record["split_path"] == str(Path(split["path"]).resolve())
    assert record["metrics"]["train_frame_count"] == 4
    assert record["metrics"]["validation_frame_count"] == 2
    assert Path(record["split_snapshot_path"]).is_file()
    assert Path(record["trainer_manifest_snapshot_path"]).is_file()
    assert len(record["split_sha256"]) == 64
    assert len(record["trainer_manifest_sha256"]) == 64
    assert Path(record["trainer_entrypoint_path"]).is_file()
    assert len(record["trainer_entrypoint_sha256"]) == 64
    assert defaults == {"max_samples": 8, "split_name": "test"}

    missing_split = services.validate_inference_setup(
        manifest,
        artifact_path=training["artifact_path"],
        dataset_root=str(dataset_root),
        adapter="orfd",
        sequence_id="training/seq_0001",
    )
    assert missing_split["ready"] is False
    assert "requires a dataset split" in " ".join(missing_split["issues"])

    inference = services.run_inference_manifest_job(
        manifest,
        artifact_path=training["artifact_path"],
        dataset_root=str(dataset_root),
        adapter="orfd",
        sequence_id="training/seq_0001",
        split_path=split["path"],
        parameters={"max_samples": 3, "split_name": "test"},
        output_dir=str(tmp_path / "inference"),
    )

    assert inference["status"] == "completed"
    assert inference["metrics"]["depth_rmse_m"] > 0.0
    assert inference["metrics"]["split_name"] == "test"
    assert len(inference["predictions"]) == 2
    assert Path(inference["previews"]["primary"]).is_file()


def test_tiny_depth_divergence_fails_without_publishing_checkpoint(tmp_path: Path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=8)
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        output_path=tmp_path / "split.json",
    )
    output = tmp_path / "diverged"

    with pytest.raises(RuntimeError, match="diverged"):
        services.run_trainer_manifest_job(
            services.CONFIG_ROOT / "trainers" / "tiny_depth.yaml",
            dataset_root=str(dataset_root),
            output_dir=str(output),
            adapter="orfd",
            split_path=split["path"],
            parameters={
                "epochs": 500,
                "learning_rate": 1.0,
                "ridge": 1.0,
                "max_frames": 6,
                "max_pixels_per_frame": 64,
                "max_depth_m": 20.0,
                "seed": 7,
            },
        )

    record = json.loads((output / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert not (output / "model.json").exists()


def test_training_snapshots_are_portable_after_sources_change(tmp_path: Path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=8)
    source_split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        output_path=tmp_path / "split.json",
    )
    first = services.run_trainer_manifest_job(
        services.CONFIG_ROOT / "trainers" / "tiny_depth.yaml",
        dataset_root=str(dataset_root),
        output_dir=str(tmp_path / "first"),
        adapter="orfd",
        split_path=source_split["path"],
        parameters={"epochs": 2, "max_frames": 6, "max_pixels_per_frame": 32},
    )
    record = json.loads(Path(first["training_run_path"]).read_text(encoding="utf-8"))
    Path(source_split["path"]).write_text('{"broken": true}', encoding="utf-8")

    rerun = services.run_trainer_manifest_job(
        record["trainer_manifest_snapshot_path"],
        dataset_root=str(dataset_root),
        output_dir=str(tmp_path / "rerun"),
        adapter="orfd",
        split_path=record["split_snapshot_path"],
        parameters={"epochs": 2, "max_frames": 6, "max_pixels_per_frame": 32},
    )

    assert Path(rerun["artifact_path"]).is_file()
    assert rerun["metrics"]["train_frame_count"] == 4

    snapshot = Path(record["split_snapshot_path"])
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 verification"):
        services.run_trainer_manifest_job(
            record["trainer_manifest_snapshot_path"],
            dataset_root=str(dataset_root),
            output_dir=str(tmp_path / "tampered"),
            adapter="orfd",
            split_path=str(snapshot),
            parameters={"epochs": 2, "max_frames": 6, "max_pixels_per_frame": 32},
        )


def test_tiny_depth_manifest_is_discovered_as_pluggable_trainer() -> None:
    row = next(item for item in services.training_preset_entries() if item["id"] == "tiny_rgb_depth")

    assert row["available"] is True
    assert row["input"]["required_modalities"] == ["front_rgb", "depth"]
    assert row["input"]["split_required"] is True
    assert row["inference"]["input"]["split_required"] is True
    assert row["inference"]["parameters"]["max_samples"]["default"] == 8
    assert row["inference"]["parameters"]["split_name"]["enum"] == ["validation", "test"]
    assert row["resume"]["supported"] is True
    assert "{resume_checkpoint}" in row["resume"]["arguments"]


def test_tiny_depth_checkpoint_can_resume_through_generic_trainer_protocol(tmp_path: Path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "dataset", frame_count=8)
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        output_path=tmp_path / "split.json",
    )
    manifest = services.CONFIG_ROOT / "trainers" / "tiny_depth.yaml"
    parameters = {"epochs": 2, "max_frames": 6, "max_pixels_per_frame": 32}
    first = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(dataset_root),
        output_dir=str(tmp_path / "first"),
        adapter="orfd",
        split_path=split["path"],
        parameters=parameters,
    )
    resumed = services.run_trainer_manifest_job(
        manifest,
        dataset_root=str(dataset_root),
        output_dir=str(tmp_path / "resumed"),
        adapter="orfd",
        split_path=split["path"],
        parameters=parameters,
        resume_checkpoint=first["artifact_path"],
    )
    record = json.loads(Path(resumed["training_run_path"]).read_text(encoding="utf-8"))

    assert resumed["metrics"]["epoch"] == 4
    assert "--resume" in resumed["command"]
    assert first["artifact_path"] in resumed["command"]
    assert record["summary"]["resume_checkpoint"] == first["artifact_path"]
