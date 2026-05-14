"""Built-in manifest-based dataset adapter used by M8 tests and examples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from offroad_sim.core import VehicleState
from offroad_sim.datasets.adapters.base import DatasetAdapter
from offroad_sim.datasets.types import DatasetFrame, DatasetSequence
from offroad_sim.utils.yaml_io import load_yaml_file


MANIFEST_NAMES = ("dataset.yaml", "dataset.yml", "manifest.yaml", "manifest.yml")


class OffroadSimV1Adapter(DatasetAdapter):
    """Adapter for the repo's small, manifest-driven dataset layout."""

    name = "offroad_sim_v1"
    priority = 10

    def can_load(self, dataset_root: str | Path) -> bool:
        root = Path(dataset_root)
        manifest = self._load_manifest(root)
        if manifest:
            adapter = str(manifest.get("adapter", ""))
            dataset_type = str(manifest.get("dataset_type", ""))
            if adapter == self.name or dataset_type == self.name:
                return True

        sequences_dir = root / str(manifest.get("sequences_dir", "sequences"))
        if not sequences_dir.is_dir():
            return False
        return any((path / "poses.csv").is_file() for path in sequences_dir.iterdir() if path.is_dir())

    def list_sequences(self, dataset_root: str | Path) -> list[str]:
        root = Path(dataset_root)
        manifest = self._load_manifest(root)
        sequences_dir = root / str(manifest.get("sequences_dir", "sequences"))
        if not sequences_dir.is_dir():
            raise FileNotFoundError(f"Dataset sequences directory not found: {sequences_dir}")

        return sorted(path.name for path in sequences_dir.iterdir() if path.is_dir())

    def load_sequence(self, dataset_root: str | Path, sequence_id: str) -> DatasetSequence:
        root = Path(dataset_root)
        manifest = self._load_manifest(root)
        sequences_dir = root / str(manifest.get("sequences_dir", "sequences"))
        sequence_dir = sequences_dir / sequence_id
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"Dataset sequence not found: {sequence_dir}")

        pose_path = sequence_dir / "poses.csv"
        if not pose_path.is_file():
            raise FileNotFoundError(f"Pose table not found: {pose_path}")

        metadata = self._load_json(sequence_dir / "metadata.json")
        calibration = self._load_json(sequence_dir / "calibration.json")
        frames = self._load_frames(sequence_dir, pose_path)
        if not frames:
            raise ValueError(f"Dataset sequence has no frames: {sequence_dir}")

        goal = metadata.get("goal")
        if isinstance(goal, list | tuple) and len(goal) >= 2:
            normalized_goal = (float(goal[0]), float(goal[1]))
        else:
            last_state = frames[-1].vehicle_state
            normalized_goal = (last_state.x, last_state.y)

        return DatasetSequence(
            dataset_id=str(manifest.get("dataset_id", root.name)),
            dataset_type=str(manifest.get("dataset_type", self.name)),
            sequence_id=sequence_id,
            root=str(root.resolve()),
            frames=frames,
            goal=normalized_goal,
            calibration=calibration,
            metadata=metadata,
        )

    def _load_frames(self, sequence_dir: Path, pose_path: Path) -> list[DatasetFrame]:
        frames: list[DatasetFrame] = []
        with pose_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                frame_id = row.get("frame_id") or row.get("id") or f"frame_{index:06d}"
                state = VehicleState(
                    x=self._as_float(row.get("x")),
                    y=self._as_float(row.get("y")),
                    z=self._as_float(row.get("z")),
                    yaw=self._as_float(row.get("yaw")),
                    pitch=self._as_float(row.get("pitch")),
                    roll=self._as_float(row.get("roll")),
                    speed=self._as_float(row.get("speed")),
                )
                frames.append(
                    DatasetFrame(
                        frame_id=frame_id,
                        timestamp=self._as_float(row.get("timestamp"), default=float(index)),
                        vehicle_state=state,
                        front_rgb_path=self._find_asset(sequence_dir, "images", frame_id, [".npy", ".png", ".jpg", ".jpeg"]),
                        depth_path=self._find_asset(sequence_dir, "depth", frame_id, [".npy", ".png"]),
                        lidar_path=self._find_asset(sequence_dir, "lidar", frame_id, [".npy"]),
                        local_bev_path=self._find_asset(sequence_dir, "bev", frame_id, [".npy"]),
                        terrain_map_path=self._find_asset(sequence_dir, "terrain", frame_id, [".npy"]),
                        label_path=self._find_asset(sequence_dir, "labels", frame_id, [".json"]),
                        metadata={"row_index": index},
                    )
                )

        return frames

    def _load_manifest(self, root: Path) -> dict[str, Any]:
        manifest_path = self._find_manifest(root)
        if manifest_path is None:
            return {}
        return load_yaml_file(manifest_path)

    def _find_manifest(self, root: Path) -> Path | None:
        for name in MANIFEST_NAMES:
            path = root / name
            if path.is_file():
                return path
        return None

    def _find_asset(self, sequence_dir: Path, directory: str, frame_id: str, suffixes: list[str]) -> str | None:
        asset_dir = sequence_dir / directory
        if not asset_dir.is_dir():
            return None
        for suffix in suffixes:
            path = asset_dir / f"{frame_id}{suffix}"
            if path.is_file():
                return str(path.resolve())
        return None

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return data

    def _as_float(self, value: str | None, default: float = 0.0) -> float:
        if value in (None, ""):
            return default
        return float(value)
