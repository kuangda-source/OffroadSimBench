"""Simulator-neutral dataset metadata types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from offroad_sim.core import Action, VehicleState


@dataclass(slots=True)
class DatasetFrame:
    """One normalized frame from any supported dataset layout."""

    frame_id: str
    timestamp: float
    vehicle_state: VehicleState
    front_rgb_path: str | None = None
    depth_path: str | None = None
    lidar_path: str | None = None
    local_bev_path: str | None = None
    terrain_map_path: str | None = None
    label_path: str | None = None
    action: Action | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def available_assets(self) -> dict[str, str]:
        """Return only the sensor/label assets present on disk."""

        assets = {
            "front_rgb": self.front_rgb_path,
            "depth": self.depth_path,
            "lidar_points": self.lidar_path,
            "local_bev": self.local_bev_path,
            "terrain_map": self.terrain_map_path,
            "label": self.label_path,
        }
        return {key: value for key, value in assets.items() if value is not None}


@dataclass(slots=True)
class DatasetSequence:
    """A normalized driving sequence independent of the original dataset format."""

    dataset_id: str
    dataset_type: str
    sequence_id: str
    root: str
    frames: list[DatasetFrame]
    goal: tuple[float, float] | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)
