from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from offroad_sim.datasets import default_dataset_registry


def test_manifest_dataset_adapter_loads_yaml_mapped_sequence(tmp_path) -> None:
    root = tmp_path / "custom_drive"
    sequence = root / "clip_001"
    (sequence / "images").mkdir(parents=True)
    (sequence / "depth").mkdir()
    (sequence / "labels").mkdir()
    for frame_id in ["000000", "000001"]:
        np.save(sequence / "images" / f"{frame_id}.npy", np.zeros((4, 5, 3), dtype=np.uint8))
        np.save(sequence / "depth" / f"{frame_id}.npy", np.zeros((4, 5), dtype=np.float32))
        np.save(sequence / "labels" / f"{frame_id}.npy", np.ones((4, 5), dtype=np.uint8))

    with (sequence / "poses.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["frame_id", "timestamp", "x", "y", "yaw", "speed"])
        writer.writeheader()
        writer.writerow({"frame_id": "000000", "timestamp": "0.0", "x": "1.0", "y": "2.0", "yaw": "0.1", "speed": "3.0"})
        writer.writerow({"frame_id": "000001", "timestamp": "0.1", "x": "1.5", "y": "2.5", "yaw": "0.2", "speed": "3.5"})

    (root / "dataset_manifest.yaml").write_text(
        """
adapter: manifest_dataset
dataset_id: custom_drive
dataset_type: camera_depth_labels
metadata:
  license: local-test
sequences:
  - id: clip_001
    root: clip_001
    pose_csv: poses.csv
    goal: [9.0, 10.0]
    assets:
      front_rgb: images/{frame_id}.npy
      depth: depth/{frame_id}.npy
      label: labels/{frame_id}.npy
""",
        encoding="utf-8",
    )

    registry = default_dataset_registry()
    adapter = registry.resolve(root, "manifest_dataset")
    loaded = adapter.load_sequence(root, "clip_001")

    assert adapter.list_sequences(root) == ["clip_001"]
    assert loaded.dataset_id == "custom_drive"
    assert loaded.dataset_type == "camera_depth_labels"
    assert loaded.goal == (9.0, 10.0)
    assert loaded.metadata["license"] == "local-test"
    assert len(loaded.frames) == 2
    assert loaded.frames[0].vehicle_state.x == 1.0
    assert loaded.frames[1].timestamp == 0.1
    assert Path(loaded.frames[0].front_rgb_path).name == "000000.npy"
    assert Path(loaded.frames[0].depth_path).name == "000000.npy"
    assert Path(loaded.frames[0].label_path).name == "000000.npy"
