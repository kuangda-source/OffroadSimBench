"""Mock dataset generator for DatasetReplayBackend development and tests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def create_mock_dataset(
    root: str | Path,
    *,
    dataset_id: str = "mock_offroad",
    sequence_id: str = "seq_0001",
    frame_count: int = 12,
    seed: int = 0,
) -> Path:
    """Create a tiny manifest-based dataset and return its root path."""

    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")

    root_path = Path(root)
    sequence_path = root_path / "sequences" / sequence_id
    for directory in ["images", "depth", "lidar", "bev", "terrain", "labels"]:
        (sequence_path / directory).mkdir(parents=True, exist_ok=True)

    (root_path / "dataset.yaml").write_text(
        "\n".join(
            [
                f"dataset_id: {dataset_id}",
                "dataset_type: offroad_sim_v1",
                "adapter: offroad_sim_v1",
                "version: 1",
                "sequences_dir: sequences",
                "",
            ]
        ),
        encoding="utf-8",
    )

    goal = ((frame_count - 1) * 1.5, math.sin((frame_count - 1) * 0.35) * 2.0)
    metadata = {
        "sequence_id": sequence_id,
        "frame_count": frame_count,
        "goal": [goal[0], goal[1]],
        "description": "Small deterministic dataset for replay backend tests.",
    }
    (sequence_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    calibration = {
        "front_camera": {"width": 8, "height": 8, "channels": 3},
        "lidar": {"frame": "base_link"},
    }
    (sequence_path / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    rng = np.random.default_rng(seed)
    pose_path = sequence_path / "poses.csv"
    with pose_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame_id", "timestamp", "x", "y", "z", "yaw", "pitch", "roll", "speed"],
        )
        writer.writeheader()
        for index in range(frame_count):
            frame_id = f"frame_{index:06d}"
            x = index * 1.5
            y = math.sin(index * 0.35) * 2.0
            yaw = math.atan2(y - math.sin(max(index - 1, 0) * 0.35) * 2.0, 1.5)
            speed = 1.5 if index else 0.0
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": f"{index * 0.1:.3f}",
                    "x": f"{x:.6f}",
                    "y": f"{y:.6f}",
                    "z": "0.000000",
                    "yaw": f"{yaw:.6f}",
                    "pitch": "0.000000",
                    "roll": "0.000000",
                    "speed": f"{speed:.6f}",
                }
            )

            rgb = np.full((8, 8, 3), index, dtype=np.uint8)
            rgb[:, :, 1] = np.arange(8, dtype=np.uint8).reshape(8, 1)
            depth = np.full((8, 8), 5.0 + index * 0.1, dtype=np.float32)
            lidar = rng.normal(loc=(x, y, 0.0), scale=0.1, size=(16, 3)).astype(np.float32)
            bev = rng.random((4, 12, 12), dtype=np.float32)
            terrain = rng.random((4, 24, 24), dtype=np.float32)
            label = {"frame_id": frame_id, "terrain_risk": float(terrain[3].mean())}

            np.save(sequence_path / "images" / f"{frame_id}.npy", rgb)
            np.save(sequence_path / "depth" / f"{frame_id}.npy", depth)
            np.save(sequence_path / "lidar" / f"{frame_id}.npy", lidar)
            np.save(sequence_path / "bev" / f"{frame_id}.npy", bev)
            np.save(sequence_path / "terrain" / f"{frame_id}.npy", terrain)
            (sequence_path / "labels" / f"{frame_id}.json").write_text(json.dumps(label), encoding="utf-8")

    return root_path


def create_mock_orfd_dataset(
    root: str | Path,
    *,
    split: str = "training",
    sequence_id: str = "seq_0001",
    frame_count: int = 8,
) -> Path:
    """Create a tiny ORFD-like dataset for adapter and phase-three tests."""

    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")

    root_path = Path(root)
    sequence_path = root_path / split / sequence_id
    for directory in ["calib", "sparse_depth", "dense_depth", "lidar_data", "image_data", "gt_image"]:
        (sequence_path / directory).mkdir(parents=True, exist_ok=True)

    (root_path / "dataset.yaml").write_text(
        "\n".join(
            [
                "dataset_id: mock_orfd",
                "dataset_type: orfd",
                "adapter: orfd",
                "version: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (sequence_path / "metadata.json").write_text(
        json.dumps({"dataset_id": "mock_orfd", "sequence_id": f"{split}/{sequence_id}"}, indent=2),
        encoding="utf-8",
    )
    (sequence_path / "calib" / "camera_lidar.txt").write_text(
        "mock calibration\nlidar_axes=x_left_y_forward_z_up\n",
        encoding="utf-8",
    )

    for index in range(frame_count):
        frame_id = f"{index:06d}"
        rgb = np.zeros((10, 12, 3), dtype=np.uint8)
        rgb[:, :, 0] = index * 8
        rgb[:, :, 1] = np.arange(12, dtype=np.uint8)
        depth = np.full((10, 12), 4.0 + index * 0.2, dtype=np.float32)
        lidar = np.column_stack(
            [
                np.linspace(-2.0, 2.0, 24),
                np.full(24, index, dtype=float),
                np.zeros(24, dtype=float),
            ]
        ).astype(np.float32)
        label = np.zeros((10, 12), dtype=np.uint8)
        label[:, 2:10] = 255

        np.save(sequence_path / "image_data" / f"{frame_id}.npy", rgb)
        np.save(sequence_path / "dense_depth" / f"{frame_id}.npy", depth)
        np.save(sequence_path / "lidar_data" / f"{frame_id}.npy", lidar)
        np.save(sequence_path / "gt_image" / f"{frame_id}.npy", label)

    return root_path
