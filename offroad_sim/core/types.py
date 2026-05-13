"""Core data structures shared by agents, backends, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ArrayLike = Any


@dataclass(slots=True)
class Action:
    """Normalized driving command used by all simulator backends."""

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Action":
        return cls(
            steer=float(data.get("steer", 0.0)),
            throttle=float(data.get("throttle", 0.0)),
            brake=float(data.get("brake", 0.0)),
        )


@dataclass(slots=True)
class VehicleState:
    """Vehicle pose and motion state in a simulator-neutral frame."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    speed: float = 0.0


@dataclass(slots=True)
class Observation:
    """Observation returned by a backend at one simulation step."""

    timestamp: float
    vehicle_state: VehicleState
    goal: tuple[float, float]
    front_rgb: ArrayLike | None = None
    depth: ArrayLike | None = None
    lidar_points: ArrayLike | None = None
    local_bev: ArrayLike | None = None
    terrain_map: ArrayLike | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    """Result returned by `OffroadSimBackend.step`."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(slots=True)
class EpisodeInfo:
    """Metadata describing one benchmark episode."""

    episode_id: str
    scenario_id: str | None = None
    vehicle_id: str | None = None
    agent_id: str | None = None
    backend: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

