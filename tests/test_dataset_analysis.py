from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from desktop_app import services
from offroad_sim.core import VehicleState
from offroad_sim.datasets import DatasetFrame, DatasetSequence, analyze_dataset_sequences, build_dataset_split


def _sequence(root: Path) -> DatasetSequence:
    frames: list[DatasetFrame] = []
    for index, frame_id in enumerate(("000000", "000001", "000003", "000004")):
        rgb = root / f"rgb_{index}.png"
        if index == 2:
            rgb.write_text("broken", encoding="utf-8")
        else:
            Image.fromarray(np.full((8, 12, 3), index * 20, dtype=np.uint8)).save(rgb)
        depth_path: str | None = None
        if index < 3:
            depth = root / f"depth_{index}.npy"
            np.save(depth, np.full((8, 12), index, dtype=np.float32))
            depth_path = str(depth)
        lidar = root / f"lidar_{index}.bin"
        np.asarray([[index, 0.0, 1.0, 0.5]], dtype=np.float32).tofile(lidar)
        frames.append(
            DatasetFrame(
                frame_id=frame_id,
                timestamp=(0.0, 0.1, 0.2, 2.0)[index],
                vehicle_state=VehicleState(x=float(index)),
                front_rgb_path=str(rgb),
                depth_path=depth_path,
                lidar_path=str(lidar),
            )
        )
    return DatasetSequence(
        dataset_id="tiny_quality",
        dataset_type="fixture",
        sequence_id="clip_001",
        root=str(root),
        frames=frames,
    )


def test_dataset_analysis_reports_stats_missing_corrupt_and_gaps(tmp_path) -> None:
    sequence = _sequence(tmp_path)

    report = analyze_dataset_sequences([sequence], dataset_root=tmp_path)

    assert report["status"] == "error"
    assert report["training_ready"] is False
    assert report["sequence_count"] == 1
    assert report["sample_count"] == 4
    assert report["modalities"] == ["depth", "front_rgb", "lidar_points"]
    assert report["missing_asset_counts"]["depth"] == 1
    assert report["corrupt_asset_count"] == 1
    assert report["resolutions"]["front_rgb"] == [[8, 12, 3]]
    assert report["resolutions"]["depth"] == [[8, 12]]
    assert report["resolutions"]["lidar_points"] == [[1, 4]]
    assert report["referenced_disk_usage_bytes"] > 0
    assert report["sequences"][0]["frame_id_gap_count"] == 1
    assert report["sequences"][0]["timestamp_issue_count"] == 1
    assert {issue["code"] for issue in report["issues"]} >= {
        "missing_asset",
        "corrupt_asset",
        "frame_id_gap",
        "timestamp_gap",
    }


def test_dataset_split_is_deterministic_and_covers_every_frame(tmp_path) -> None:
    sequence = _sequence(tmp_path)

    first = build_dataset_split(
        [sequence],
        dataset_root=tmp_path,
        adapter="fixture",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=9,
    )
    second = build_dataset_split(
        [sequence],
        dataset_root=tmp_path,
        adapter="fixture",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=9,
    )

    assert first == second
    assert first["split_unit"] == "frame"
    assert first["strategy"] == "contiguous_frame_ranges"
    assert first["seed_applied"] is False
    assert first["counts"] == {"train": 2, "validation": 1, "test": 1}
    indices = {
        index
        for rows in first["splits"].values()
        for row in rows
        for index in row["frame_indices"]
    }
    assert indices == {0, 1, 2, 3}
    split_sets = [
        {index for row in first["splits"][name] for index in row["frame_indices"]}
        for name in ("train", "validation", "test")
    ]
    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])


def test_dataset_quality_service_writes_json_and_markdown_reports(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    services.create_mock_orfd_dataset(dataset_root, split="training", sequence_id="seq_0001", frame_count=4)
    output_dir = tmp_path / "report"

    payload = services.analyze_dataset_quality(
        str(dataset_root),
        "orfd",
        output_dir=output_dir,
    )
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "orfd",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        output_path=tmp_path / "split.json",
    )

    report = json.loads(Path(payload["report_json_path"]).read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["training_ready"] is True
    assert report["sample_count"] == 4
    assert "Dataset Quality Report" in Path(payload["report_markdown_path"]).read_text(encoding="utf-8")
    assert split["counts"] == {"train": 2, "validation": 1, "test": 1}
    assert Path(split["path"]).is_file()


def test_dataset_preview_generates_lidar_image(tmp_path) -> None:
    dataset_root = tmp_path / "manifest"
    sequence_root = dataset_root / "clip"
    sequence_root.mkdir(parents=True)
    (sequence_root / "poses.csv").write_text("frame_id,timestamp,x,y\n000000,0,0,0\n", encoding="utf-8")
    np.asarray([[0.0, 0.0, 1.0, 0.5], [2.0, 1.0, 0.0, 0.4]], dtype=np.float32).tofile(
        sequence_root / "000000.bin"
    )
    (dataset_root / "dataset_manifest.yaml").write_text(
        """
adapter: manifest_dataset
dataset_id: lidar_fixture
sequences:
  - id: clip
    root: clip
    pose_csv: poses.csv
    assets:
      lidar: "{frame_id}.bin"
""",
        encoding="utf-8",
    )

    payload = services.preview_dataset_frame(
        str(dataset_root),
        "manifest_dataset",
        "clip",
        output_dir=tmp_path / "previews",
    )

    assert Path(payload["previews"]["lidar_points"]).is_file()


def test_manifest_quality_reports_modality_when_every_asset_is_missing(tmp_path) -> None:
    dataset_root = tmp_path / "missing_manifest"
    sequence_root = dataset_root / "clip"
    sequence_root.mkdir(parents=True)
    (sequence_root / "poses.csv").write_text("frame_id,timestamp,x,y\n000000,0,0,0\n", encoding="utf-8")
    (dataset_root / "dataset_manifest.yaml").write_text(
        """
adapter: manifest_dataset
dataset_id: missing_fixture
sequences:
  - id: clip
    root: clip
    pose_csv: poses.csv
    assets:
      label: labels/{frame_id}.png
""",
        encoding="utf-8",
    )

    payload = services.analyze_dataset_quality(
        str(dataset_root),
        "manifest_dataset",
        output_dir=tmp_path / "missing_report",
    )

    assert payload["status"] == "error"
    assert payload["analysis"]["modalities"] == ["label"]
    assert payload["analysis"]["available_modalities"] == []
    assert payload["analysis"]["missing_asset_counts"] == {"label": 1}


def test_manifest_dataset_end_to_end_quality_split_preview_and_tiny_training(tmp_path) -> None:
    dataset_root = tmp_path / "multimodal_drive"
    sequence_root = dataset_root / "clip_001"
    for name in ("rgb", "depth", "lidar"):
        (sequence_root / name).mkdir(parents=True, exist_ok=True)
    pose_rows = ["frame_id,timestamp,x,y,yaw,speed"]
    for index in range(5):
        frame_id = f"{index:06d}"
        pose_rows.append(f"{frame_id},{index * 0.1},{index * 0.5},0,0,5")
        Image.fromarray(np.full((10, 14, 3), index * 30, dtype=np.uint8)).save(
            sequence_root / "rgb" / f"{frame_id}.png"
        )
        np.save(sequence_root / "depth" / f"{frame_id}.npy", np.full((10, 14), index + 1, dtype=np.float32))
        np.asarray([[index, 0.0, 1.0, 0.5], [index + 1, 1.0, 0.5, 0.4]], dtype=np.float32).tofile(
            sequence_root / "lidar" / f"{frame_id}.bin"
        )
    (sequence_root / "poses.csv").write_text("\n".join(pose_rows) + "\n", encoding="utf-8")
    (dataset_root / "dataset_manifest.yaml").write_text(
        """
adapter: manifest_dataset
dataset_id: multimodal_drive
dataset_type: tiny_multimodal
sequences:
  - id: clip_001
    root: clip_001
    pose_csv: poses.csv
    assets:
      front_rgb: rgb/{frame_id}.png
      depth: depth/{frame_id}.npy
      lidar: lidar/{frame_id}.bin
""",
        encoding="utf-8",
    )

    inspection = services.inspect_dataset(str(dataset_root), "manifest_dataset", "clip_001")
    quality = services.analyze_dataset_quality(
        str(dataset_root),
        "manifest_dataset",
        output_dir=tmp_path / "quality",
    )
    split = services.create_dataset_split_definition(
        str(dataset_root),
        "manifest_dataset",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        output_path=tmp_path / "split.json",
    )
    previews = [
        services.preview_dataset_frame(
            str(dataset_root),
            "manifest_dataset",
            "clip_001",
            frame_index=index,
            output_dir=tmp_path / "previews",
        )
        for index in range(5)
    ]
    training = services.train_tiny_world_model(
        str(dataset_root),
        str(tmp_path / "model"),
        adapter="manifest_dataset",
        sequence_id="clip_001",
    )

    assert inspection["details"]["analysis_scope"] == "selected_sequence"
    assert inspection["details"]["dataset_sequence_count"] == 1
    assert quality["analysis"]["asset_check_mode"] == "full"
    assert quality["analysis"]["checked_asset_count"] == 15
    assert quality["analysis"]["modalities"] == ["depth", "front_rgb", "lidar_points"]
    assert quality["training_ready"] is True
    assert split["counts"] == {"train": 3, "validation": 1, "test": 1}
    assert split["strategy"] == "contiguous_frame_ranges"
    assert [payload["frame_index"] for payload in previews] == list(range(5))
    assert all(set(payload["previews"]) == {"front_rgb", "depth", "lidar_points"} for payload in previews)
    assert all(Path(path).is_file() for payload in previews for path in payload["previews"].values())
    assert Path(training["model_path"]).is_file()
    training_run = json.loads(Path(training["training_run_path"]).read_text(encoding="utf-8"))
    assert training_run["history"]["train_rmse"]
