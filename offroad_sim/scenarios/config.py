"""Scenario configuration models and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from offroad_sim.utils.yaml_io import load_yaml_file


def _tuple2(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    items = list(value)
    if len(items) != 2:
        raise ValueError("Expected a 2-value sequence")
    return (float(items[0]), float(items[1]))


@dataclass(slots=True)
class TerrainConfig:
    type: str
    difficulty: str = "medium"
    risk_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TerrainConfig":
        data = data or {}
        return cls(
            type=str(data.get("type", "unknown")),
            difficulty=str(data.get("difficulty", "medium")),
            risk_scale=float(data.get("risk_scale", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class TaskConfig:
    max_time_sec: float = 180.0
    success_radius_m: float = 5.0
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (50.0, 50.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskConfig":
        data = data or {}
        return cls(
            max_time_sec=float(data.get("max_time_sec", 180.0)),
            success_radius_m=float(data.get("success_radius_m", 5.0)),
            start=_tuple2(data.get("start"), (0.0, 0.0)),
            goal=_tuple2(data.get("goal"), (50.0, 50.0)),
        )


@dataclass(slots=True)
class MetricsConfig:
    collision: bool = True
    rollover: bool = True
    path_length: bool = True
    terrain_risk: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MetricsConfig":
        data = data or {}
        return cls(
            collision=bool(data.get("collision", True)),
            rollover=bool(data.get("rollover", True)),
            path_length=bool(data.get("path_length", True)),
            terrain_risk=bool(data.get("terrain_risk", True)),
        )


@dataclass(slots=True)
class ScenarioConfig:
    scenario_id: str
    backend: str
    map: str
    weather: str
    terrain: TerrainConfig
    task: TaskConfig
    metrics: MetricsConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioConfig":
        return cls(
            scenario_id=str(data["scenario_id"]),
            backend=str(data["backend"]),
            map=str(data["map"]),
            weather=str(data.get("weather", "clear")),
            terrain=TerrainConfig.from_dict(data.get("terrain")),
            task=TaskConfig.from_dict(data.get("task")),
            metrics=MetricsConfig.from_dict(data.get("metrics")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        return cls.from_dict(load_yaml_file(path))


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    return ScenarioConfig.from_yaml(path)

