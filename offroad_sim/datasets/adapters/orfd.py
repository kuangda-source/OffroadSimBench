"""Adapter for the ORFD off-road freespace dataset layout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from offroad_sim.core import VehicleState
from offroad_sim.datasets.adapters.base import DatasetAdapter
from offroad_sim.datasets.types import DatasetFrame, DatasetSequence
from offroad_sim.utils.yaml_io import load_yaml_file


SPLITS = ("training", "validation", "testing", "train", "val", "test")
IMAGE_DIRS = ("image_data", "images", "rgb")
GT_DIRS = ("gt_image", "gt", "labels", "mask")
DEPTH_DIRS = ("dense_depth", "sparse_depth", "depth")
LIDAR_DIRS = ("lidar_data", "lidar")
CALIB_DIRS = ("calib", "calibration")
MANIFEST_NAMES = ("dataset.yaml", "dataset.yml", "manifest.yaml", "manifest.yml")


class ORFDAdapter(DatasetAdapter):
    """Read ORFD sequences into the simulator-neutral dataset interface.

    ORFD focuses on off-road freespace perception and commonly ships images,
    depth/lidar assets, calibration, and ground-truth masks rather than vehicle
    control logs. When no pose table is present, this adapter creates a
    deterministic index-order trajectory and marks it in frame metadata. That
    keeps the dataset usable for perception-conditioned smoke tests while making
    the synthetic pose assumption explicit.
    """

    name = "orfd"
    priority = 20

    def can_load(self, dataset_root: str | Path) -> bool:
        root = Path(dataset_root)
        manifest = self._load_manifest(root)
        if manifest.get("adapter") == self.name or manifest.get("dataset_type") == self.name:
            return True
        return any((root / split).is_dir() for split in SPLITS)

    def list_sequences(self, dataset_root: str | Path) -> list[str]:
        root = Path(dataset_root)
        rows: list[str] = []
        for split in SPLITS:
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                if self._find_asset_dir(sequence_dir, IMAGE_DIRS) is not None:
                    rows.append(f"{split}/{sequence_dir.name}")
        if not rows and self._find_asset_dir(root, IMAGE_DIRS) is not None:
            rows.append(root.name)
        if not rows:
            raise FileNotFoundError(f"No ORFD sequences found under {root}")
        return rows

    def load_sequence(self, dataset_root: str | Path, sequence_id: str) -> DatasetSequence:
        root = Path(dataset_root)
        sequence_dir = self._sequence_dir(root, sequence_id)
        image_dir = self._find_asset_dir(sequence_dir, IMAGE_DIRS)
        if image_dir is None:
            raise FileNotFoundError(f"ORFD image directory not found in {sequence_dir}")

        frame_ids = self._frame_ids(image_dir)
        poses = self._load_pose_table(sequence_dir)
        metadata = self._load_json(sequence_dir / "metadata.json")
        calibration = self._load_calibration(sequence_dir)

        frames: list[DatasetFrame] = []
        for index, frame_id in enumerate(frame_ids):
            state = poses.get(frame_id) or self._synthetic_state(index)
            pose_source = "poses.csv" if frame_id in poses else "synthetic_index_order"
            frames.append(
                DatasetFrame(
                    frame_id=frame_id,
                    timestamp=float(index),
                    vehicle_state=state,
                    front_rgb_path=self._find_asset(sequence_dir, IMAGE_DIRS, frame_id),
                    depth_path=self._find_asset(sequence_dir, DEPTH_DIRS, frame_id),
                    lidar_path=self._find_asset(sequence_dir, LIDAR_DIRS, frame_id),
                    label_path=self._find_asset(sequence_dir, GT_DIRS, frame_id),
                    metadata={
                        "row_index": index,
                        "pose_source": pose_source,
                        "orfd_sequence_dir": str(sequence_dir.resolve()),
                    },
                )
            )

        if not frames:
            raise ValueError(f"ORFD sequence has no frames: {sequence_dir}")

        last_state = frames[-1].vehicle_state
        return DatasetSequence(
            dataset_id=str(metadata.get("dataset_id", "orfd")),
            dataset_type=self.name,
            sequence_id=sequence_id,
            root=str(root.resolve()),
            frames=frames,
            goal=(last_state.x, last_state.y),
            calibration=calibration,
            metadata={
                **metadata,
                "adapter": self.name,
                "source_layout": "orfd",
                "frame_count": len(frames),
            },
        )

    def _sequence_dir(self, root: Path, sequence_id: str) -> Path:
        direct = root / sequence_id
        if direct.is_dir():
            return direct
        parts = Path(sequence_id).parts
        if len(parts) == 1 and self._find_asset_dir(root, IMAGE_DIRS) is not None:
            return root
        candidate = root.joinpath(*parts)
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"ORFD sequence not found: {sequence_id}")

    def _frame_ids(self, image_dir: Path) -> list[str]:
        suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}
        return sorted(path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)

    def _load_pose_table(self, sequence_dir: Path) -> dict[str, VehicleState]:
        path = sequence_dir / "poses.csv"
        if not path.is_file():
            return {}
        rows: dict[str, VehicleState] = {}
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                frame_id = row.get("frame_id") or row.get("id") or f"frame_{index:06d}"
                rows[frame_id] = VehicleState(
                    x=self._as_float(row.get("x")),
                    y=self._as_float(row.get("y")),
                    z=self._as_float(row.get("z")),
                    yaw=self._as_float(row.get("yaw")),
                    pitch=self._as_float(row.get("pitch")),
                    roll=self._as_float(row.get("roll")),
                    speed=self._as_float(row.get("speed")),
                )
        return rows

    def _synthetic_state(self, index: int) -> VehicleState:
        return VehicleState(x=0.0, y=float(index), z=0.0, yaw=math.pi / 2.0, speed=1.0 if index else 0.0)

    def _find_asset_dir(self, sequence_dir: Path, names: tuple[str, ...]) -> Path | None:
        for name in names:
            path = sequence_dir / name
            if path.is_dir():
                return path
        return None

    def _find_asset(self, sequence_dir: Path, dirs: tuple[str, ...], frame_id: str) -> str | None:
        directory = self._find_asset_dir(sequence_dir, dirs)
        if directory is None:
            return None
        for suffix in (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".bin", ".pcd"):
            path = directory / f"{frame_id}{suffix}"
            if path.is_file():
                return str(path.resolve())
        return None

    def _load_manifest(self, root: Path) -> dict[str, Any]:
        for name in MANIFEST_NAMES:
            path = root / name
            if path.is_file():
                return load_yaml_file(path)
        return {}

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def _load_calibration(self, sequence_dir: Path) -> dict[str, Any]:
        calibration: dict[str, Any] = {
            "coordinate_frame": {
                "source": "orfd",
                "lidar_axes": "x_left_y_forward_z_up",
            }
        }
        for directory_name in CALIB_DIRS:
            directory = sequence_dir / directory_name
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    calibration[path.name] = str(path.resolve())
        return calibration

    def _as_float(self, value: str | None, default: float = 0.0) -> float:
        if value in (None, ""):
            return default
        return float(value)
