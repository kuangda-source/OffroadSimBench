"""Vehicle configuration models and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from offroad_sim.utils.yaml_io import load_yaml_file


def _tuple3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    items = list(value)
    if len(items) != 3:
        raise ValueError("Expected a 3-value sequence")
    return (float(items[0]), float(items[1]), float(items[2]))


@dataclass(slots=True)
class SensorConfig:
    sensor_id: str
    sensor_type: str = "generic"
    enabled: bool = True
    update_rate_hz: float = 10.0
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorConfig":
        sensor_type = str(data.get("sensor_type", data.get("type", "generic")))
        sensor_id = str(data.get("sensor_id", data.get("id", sensor_type)))
        common = {
            "sensor_id": sensor_id,
            "enabled": bool(data.get("enabled", True)),
            "update_rate_hz": float(data.get("update_rate_hz", 10.0)),
            "mount_xyz": _tuple3(data.get("mount_xyz"), (0.0, 0.0, 0.0)),
            "mount_rpy": _tuple3(data.get("mount_rpy"), (0.0, 0.0, 0.0)),
        }

        if sensor_type == "camera":
            return CameraConfig.from_dict(data, **common)
        if sensor_type == "lidar":
            return LidarConfig.from_dict(data, **common)
        if sensor_type == "imu":
            return ImuConfig.from_dict(data, **common)
        if sensor_type == "gps":
            return GpsConfig.from_dict(data, **common)

        return cls(sensor_type=sensor_type, **common)


@dataclass(slots=True)
class CameraConfig(SensorConfig):
    sensor_type: str = "camera"
    width: int = 640
    height: int = 480
    fov_deg: float = 90.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], **common: Any) -> "CameraConfig":
        return cls(
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
            fov_deg=float(data.get("fov_deg", 90.0)),
            **common,
        )


@dataclass(slots=True)
class LidarConfig(SensorConfig):
    sensor_type: str = "lidar"
    channels: int = 16
    range_m: float = 80.0
    points_per_second: int = 100_000

    @classmethod
    def from_dict(cls, data: dict[str, Any], **common: Any) -> "LidarConfig":
        return cls(
            channels=int(data.get("channels", 16)),
            range_m=float(data.get("range_m", 80.0)),
            points_per_second=int(data.get("points_per_second", 100_000)),
            **common,
        )


@dataclass(slots=True)
class ImuConfig(SensorConfig):
    sensor_type: str = "imu"
    noise_std: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], **common: Any) -> "ImuConfig":
        return cls(noise_std=float(data.get("noise_std", 0.0)), **common)


@dataclass(slots=True)
class GpsConfig(SensorConfig):
    sensor_type: str = "gps"
    position_noise_m: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], **common: Any) -> "GpsConfig":
        return cls(position_noise_m=float(data.get("position_noise_m", 0.0)), **common)


@dataclass(slots=True)
class VehicleConfig:
    vehicle_id: str
    template: str
    mass_kg: float
    length_m: float
    width_m: float
    wheelbase_m: float
    max_speed_mps: float
    max_steer_deg: float
    tire_type: str
    sensors: list[SensorConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VehicleConfig":
        sensors = [SensorConfig.from_dict(item) for item in data.get("sensors", [])]
        return cls(
            vehicle_id=str(data["vehicle_id"]),
            template=str(data["template"]),
            mass_kg=float(data["mass_kg"]),
            length_m=float(data["length_m"]),
            width_m=float(data["width_m"]),
            wheelbase_m=float(data["wheelbase_m"]),
            max_speed_mps=float(data["max_speed_mps"]),
            max_steer_deg=float(data["max_steer_deg"]),
            tire_type=str(data["tire_type"]),
            sensors=sensors,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleConfig":
        return cls.from_dict(load_yaml_file(path))


def load_vehicle_config(path: str | Path) -> VehicleConfig:
    return VehicleConfig.from_yaml(path)

