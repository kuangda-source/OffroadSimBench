"""Generic YAML-manifest dataset adapter."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from offroad_sim.core import VehicleState
from offroad_sim.datasets.adapters.base import DatasetAdapter
from offroad_sim.datasets.types import DatasetFrame, DatasetSequence
from offroad_sim.utils.yaml_io import load_yaml_file


MANIFEST_NAMES = ("dataset_manifest.yaml", "dataset_manifest.yml")
ASSET_FIELDS = {
    "front_rgb": "front_rgb_path",
    "rgb": "front_rgb_path",
    "camera": "front_rgb_path",
    "depth": "depth_path",
    "lidar": "lidar_path",
    "lidar_points": "lidar_path",
    "local_bev": "local_bev_path",
    "terrain_map": "terrain_map_path",
    "label": "label_path",
    "mask": "label_path",
}


class ManifestDatasetAdapter(DatasetAdapter):
    """Read user-described driving datasets through a small YAML manifest.

    The manifest keeps arbitrary dataset layouts import-safe by declaring where
    poses and frame assets live. It is intentionally conservative: values that
    are not present are left as ``None`` instead of guessed.
    """

    name = "manifest_dataset"
    priority = 10

    def can_load(self, dataset_root: str | Path) -> bool:
        manifest = self._load_manifest(Path(dataset_root))
        return manifest.get("adapter") == self.name

    def list_sequences(self, dataset_root: str | Path) -> list[str]:
        manifest = self._require_manifest(Path(dataset_root))
        sequences = manifest.get("sequences")
        if not isinstance(sequences, list):
            raise ValueError("Manifest dataset requires a list under 'sequences'.")
        rows = [str(row.get("id") or row.get("sequence_id") or "") for row in sequences if isinstance(row, dict)]
        result = [item for item in rows if item]
        if not result:
            raise ValueError("Manifest dataset has no sequence ids.")
        return result

    def load_sequence(self, dataset_root: str | Path, sequence_id: str) -> DatasetSequence:
        root = Path(dataset_root)
        manifest = self._require_manifest(root)
        row = self._sequence_row(manifest, sequence_id)
        sequence_root = (root / str(row.get("root", "."))).resolve()
        pose_path = sequence_root / str(row.get("pose_csv", "poses.csv"))
        frames = self._load_frames(sequence_root, pose_path, row)
        if not frames:
            raise ValueError(f"Manifest sequence has no frames: {sequence_id}")

        metadata = dict(manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {})
        if isinstance(row.get("metadata"), dict):
            metadata.update(row["metadata"])
        metadata.update({"adapter": self.name, "manifest_path": str(self._manifest_path(root).resolve())})

        return DatasetSequence(
            dataset_id=str(manifest.get("dataset_id") or root.name),
            dataset_type=str(manifest.get("dataset_type") or self.name),
            sequence_id=sequence_id,
            root=str(root.resolve()),
            frames=frames,
            goal=self._goal(row, frames),
            calibration=dict(manifest.get("calibration") if isinstance(manifest.get("calibration"), dict) else {}),
            metadata=metadata,
        )

    def _load_frames(self, sequence_root: Path, pose_path: Path, sequence_row: dict[str, Any]) -> list[DatasetFrame]:
        if not pose_path.is_file():
            raise FileNotFoundError(f"Manifest pose CSV not found: {pose_path}")
        assets = sequence_row.get("assets") if isinstance(sequence_row.get("assets"), dict) else {}
        frames: list[DatasetFrame] = []
        with pose_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                frame_id = str(row.get("frame_id") or row.get("id") or f"{index:06d}")
                fields = self._asset_paths(sequence_root, assets, frame_id, index, row)
                frames.append(
                    DatasetFrame(
                        frame_id=frame_id,
                        timestamp=self._as_float(row.get("timestamp"), float(index)),
                        vehicle_state=VehicleState(
                            x=self._as_float(row.get("x")),
                            y=self._as_float(row.get("y")),
                            z=self._as_float(row.get("z")),
                            yaw=self._as_float(row.get("yaw")),
                            pitch=self._as_float(row.get("pitch")),
                            roll=self._as_float(row.get("roll")),
                            speed=self._as_float(row.get("speed")),
                        ),
                        metadata={"row_index": index, **{f"pose_{key}": value for key, value in row.items()}},
                        **fields,
                    )
                )
        return frames

    def _asset_paths(
        self,
        sequence_root: Path,
        assets: dict[str, Any],
        frame_id: str,
        index: int,
        row: dict[str, Any],
    ) -> dict[str, str | None]:
        fields: dict[str, str | None] = {}
        context = {"frame_id": frame_id, "index": index, **row}
        for asset_name, template in assets.items():
            field_name = ASSET_FIELDS.get(str(asset_name))
            if field_name is None or template in (None, ""):
                continue
            rendered = str(template).format_map(_SafeFormatDict(context))
            path = (sequence_root / rendered).resolve()
            fields[field_name] = str(path) if path.exists() else None
        return fields

    def _goal(self, row: dict[str, Any], frames: list[DatasetFrame]) -> tuple[float, float] | None:
        value = row.get("goal")
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (float(value[0]), float(value[1]))
        if frames:
            state = frames[-1].vehicle_state
            return (state.x, state.y)
        return None

    def _sequence_row(self, manifest: dict[str, Any], sequence_id: str) -> dict[str, Any]:
        sequences = manifest.get("sequences")
        if not isinstance(sequences, list):
            raise ValueError("Manifest dataset requires a list under 'sequences'.")
        for row in sequences:
            if isinstance(row, dict) and str(row.get("id") or row.get("sequence_id") or "") == sequence_id:
                return row
        raise KeyError(f"Manifest sequence not found: {sequence_id}")

    def _load_manifest(self, root: Path) -> dict[str, Any]:
        path = self._manifest_path(root)
        return load_yaml_file(path) if path.is_file() else {}

    def _require_manifest(self, root: Path) -> dict[str, Any]:
        manifest = self._load_manifest(root)
        if not manifest:
            raise FileNotFoundError(f"Manifest dataset file not found in {root}")
        return manifest

    def _manifest_path(self, root: Path) -> Path:
        for name in MANIFEST_NAMES:
            path = root / name
            if path.is_file():
                return path
        return root / MANIFEST_NAMES[0]

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        if value in (None, ""):
            return default
        return float(value)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
