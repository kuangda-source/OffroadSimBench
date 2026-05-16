"""Optional BeamNG.tech backend adapter.

The module is import-safe on machines without BeamNG.tech or beamngpy. Real
connection work is delayed until ``connect``/``reset``.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from offroad_sim.backends.base import OffroadSimBackend
from offroad_sim.backends.registry import BackendStatus
from offroad_sim.core import Action, Observation, StepResult, VehicleState
from offroad_sim.scenarios import ScenarioConfig, scenario_metadata_section
from offroad_sim.vehicles import VehicleConfig


class BeamNGUnavailableError(RuntimeError):
    """Raised when the optional BeamNG runtime cannot be used."""


@dataclass(slots=True)
class BeamNGConnectionConfig:
    """Connection and launch options for BeamNG.tech."""

    host: str = "localhost"
    port: int = 64256
    bng_home: str | None = None
    user_dir: str | None = None
    launch: bool = True
    level: str = "west_coast_usa"
    scenario_name: str = "offroad_sim_bench"
    vehicle_model: str = "pickup"
    vehicle_id: str = "ego"
    steps_per_action: int = 6
    gfx: str | None = "vk"


class BeamNGBackend(OffroadSimBackend):
    """BeamNG backend skeleton behind the shared OffroadSimBackend API."""

    def __init__(
        self,
        *,
        connection: BeamNGConnectionConfig | None = None,
        vehicle_config: VehicleConfig | None = None,
        auto_connect: bool = False,
    ) -> None:
        self.connection = connection or BeamNGConnectionConfig()
        self.vehicle_config = vehicle_config
        self._bng: Any = None
        self._scenario: Any = None
        self._vehicle: Any = None
        self._sensors: dict[str, Any] = {}
        self._sensor_cache: dict[str, Any] = {}
        self._last_observation: Observation | None = None
        self._scenario_config: Any = None
        self._step_count = 0
        self._connected = False
        self._active_level = self.connection.level
        self._active_steps_per_action = self.connection.steps_per_action
        self._route: list[tuple[float, float]] = []
        self._distance_traveled = 0.0
        self._last_position: tuple[float, float, float] | None = None
        self._collision_count = 0
        self._last_damage = 0.0
        if auto_connect:
            self.connect()

    @classmethod
    def runtime_status(cls, bng_home: str | Path | None = None) -> BackendStatus:
        beamngpy_available = importlib.util.find_spec("beamngpy") is not None
        env_home = os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME")
        resolved_home = cls._resolve_bng_home(bng_home)
        executable = cls._find_executable(resolved_home) if resolved_home is not None else None
        available = bool(beamngpy_available and executable is not None)

        missing: list[str] = []
        if not beamngpy_available:
            missing.append("beamngpy Python package")
        if resolved_home is None:
            missing.append("BNG_HOME")
        elif executable is None:
            missing.append("BeamNG.tech executable under BNG_HOME")

        message = "BeamNG backend is ready." if available else "Missing " + ", ".join(missing) + "."
        return BackendStatus(
            name="beamng",
            available=available,
            message=message,
            details={
                "beamngpy_available": beamngpy_available,
                "bng_home": str(resolved_home) if resolved_home is not None else None,
                "executable": str(executable) if executable is not None else None,
                "auto_detected_home": bool(resolved_home is not None and bng_home is None and env_home is None),
                "install_hint": "python -m pip install beamngpy",
            },
        )

    def connect(self) -> None:
        status = self.runtime_status(self.connection.bng_home)
        if not status.available:
            raise BeamNGUnavailableError(self._missing_runtime_message(status))

        beamngpy = importlib.import_module("beamngpy")
        BeamNGpy = getattr(beamngpy, "BeamNGpy")
        kwargs = {"home": str(self._resolve_bng_home(self.connection.bng_home))}
        if self.connection.user_dir:
            kwargs["user"] = str(Path(self.connection.user_dir))
        if self.connection.gfx:
            kwargs["gfx"] = self.connection.gfx
        self._bng = BeamNGpy(self.connection.host, self.connection.port, **kwargs)
        self._bng.open(launch=self.connection.launch)
        self._connected = True

    def load_scenario(self, scenario_config: ScenarioConfig | Mapping[str, Any] | None = None) -> None:
        self._ensure_connected()
        beamngpy = importlib.import_module("beamngpy")
        Scenario = getattr(beamngpy, "Scenario")
        level = self._beamng_level_for_config(scenario_config)
        self._active_level = level
        scenario_name = self._beamng_scenario_name_for_config(scenario_config)
        self._scenario = Scenario(level, scenario_name)

    def spawn_vehicle(self, vehicle_config: VehicleConfig | None = None) -> None:
        self._ensure_connected()
        beamngpy = importlib.import_module("beamngpy")
        Vehicle = getattr(beamngpy, "Vehicle")
        config = vehicle_config or self.vehicle_config
        model = self._beamng_vehicle_model_for_config(self._scenario_config, config)
        try:
            self._vehicle = Vehicle(self.connection.vehicle_id, model=model, license="OSB")
        except TypeError:
            self._vehicle = Vehicle(self.connection.vehicle_id, model=model, licence="OSB")
        if self._scenario is not None:
            pos, rot_quat = self._beamng_vehicle_start_for_config(self._scenario_config)
            self._scenario.add_vehicle(self._vehicle, pos=pos, rot_quat=rot_quat)

    def attach_sensors(self, vehicle_config: VehicleConfig | None = None) -> None:
        self._ensure_connected()
        self.vehicle_config = vehicle_config or self.vehicle_config
        self._sensors = {}
        if self._vehicle is None or self.vehicle_config is None:
            return

        try:
            sensors_module = importlib.import_module("beamngpy.sensors")
        except ImportError:
            return

        for sensor in self.vehicle_config.sensors:
            if not sensor.enabled:
                continue
            instance = self._make_sensor(sensors_module, sensor)
            if instance is None:
                continue
            attach = getattr(self._vehicle, "attach_sensor", None)
            if callable(attach):
                attach(sensor.sensor_id, instance)
            self._sensors[sensor.sensor_id] = instance

    def reset(self, scenario_config: Any = None) -> Observation:
        self._scenario_config = scenario_config
        beamng_options = self._beamng_metadata(scenario_config)
        self._active_steps_per_action = int(beamng_options.get("steps_per_action", self.connection.steps_per_action))
        self._route = self._beamng_route_for_config(scenario_config)
        self._distance_traveled = 0.0
        self._last_position = None
        self._collision_count = 0
        self._last_damage = 0.0
        if not self._connected:
            self.connect()
        if self._scenario is None:
            self.load_scenario(scenario_config)
        if self._vehicle is None:
            self.spawn_vehicle(self.vehicle_config)
            self.attach_sensors(self.vehicle_config)

        if hasattr(self._scenario, "make"):
            self._scenario.make(self._bng)
        scenario_api = getattr(self._bng, "scenario", None)
        if scenario_api is not None and hasattr(scenario_api, "load"):
            scenario_api.load(self._scenario)
        elif hasattr(self._bng, "load_scenario"):
            self._bng.load_scenario(self._scenario)
        if scenario_api is not None and hasattr(scenario_api, "start"):
            scenario_api.start()
        elif hasattr(self._bng, "start_scenario"):
            self._bng.start_scenario()
        self._configure_visible_helpers()

        self._step_count = 0
        self._last_observation = self._build_placeholder_observation(scenario_config)
        self._update_motion_metrics(self._last_observation.vehicle_state)
        return self._last_observation

    def step(self, action: Action) -> StepResult:
        self._ensure_connected()
        if self._vehicle is not None and hasattr(self._vehicle, "control"):
            try:
                self._vehicle.control(
                    throttle=action.throttle,
                    steering=action.steer,
                    brake=action.brake,
                    parkingbrake=0.0,
                    clutch=0.0,
                    gear=1,
                )
            except TypeError:
                self._vehicle.control(throttle=action.throttle, steering=action.steer, brake=action.brake)
        if self._bng is not None and hasattr(self._bng, "step"):
            self._bng.step(self._active_steps_per_action)

        self._step_count += 1
        self._sensor_cache = self._poll_sensors()
        self._last_observation = self._build_placeholder_observation(
            self._scenario_config,
            timestamp=float(self._step_count),
        )
        self._configure_visible_camera(self._last_observation.vehicle_state)
        self._update_motion_metrics(self._last_observation.vehicle_state)
        return StepResult(
            observation=self._last_observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"backend": "beamng", "step_count": self._step_count},
        )

    def get_observation(self) -> Observation:
        if self._last_observation is None:
            raise RuntimeError("BeamNGBackend has not been reset.")
        return self._last_observation

    def get_metrics(self) -> dict[str, Any]:
        return {
            "backend": "beamng",
            "connected": self._connected,
            "episode_length": self._step_count,
            "vehicle_id": self.connection.vehicle_id,
            "level": self._active_level,
            "sensor_count": len(self._sensors),
            "sensor_ids": sorted(self._sensors),
            "route_waypoint_count": len(self._route),
            "distance_traveled": self._distance_traveled,
            "collision_count": self._collision_count,
            "damage": self._last_damage,
        }

    def close(self) -> None:
        if self._bng is not None and hasattr(self._bng, "close"):
            self._bng.close()
        self._bng = None
        self._scenario = None
        self._vehicle = None
        self._sensors = {}
        self._sensor_cache = {}
        self._last_observation = None
        self._scenario_config = None
        self._connected = False
        self._step_count = 0
        self._active_level = self.connection.level
        self._active_steps_per_action = self.connection.steps_per_action
        self._route = []
        self._distance_traveled = 0.0
        self._last_position = None
        self._collision_count = 0
        self._last_damage = 0.0

    @classmethod
    def _resolve_bng_home(cls, bng_home: str | Path | None = None) -> Path | None:
        value = bng_home or os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME")
        if value:
            return Path(value)
        return cls._auto_detect_bng_home()

    @classmethod
    def _auto_detect_bng_home(cls) -> Path | None:
        repo_root = Path(__file__).resolve().parents[2]
        roots = [
            repo_root / "BeamNG",
            repo_root.parent / "BeamNG",
        ]
        for root in roots:
            if not root.exists():
                continue
            direct_candidates = [
                root / "BeamNG.tech",
                root / "BeamNG.drive",
                root / "BeamNG.tech.v0.38.3.0",
            ]
            wildcard_candidates = sorted(
                [*root.glob("BeamNG.tech*"), *root.glob("BeamNG.drive*")],
                reverse=True,
            )
            for candidate in [*direct_candidates, *wildcard_candidates]:
                if candidate.is_dir() and cls._find_executable(candidate) is not None:
                    return candidate
        return None

    @classmethod
    def _find_executable(cls, bng_home: Path | None) -> Path | None:
        if bng_home is None:
            return None
        candidates = [
            bng_home / "BeamNG.tech.exe",
            bng_home / "Bin64" / "BeamNG.tech.x64.exe",
            bng_home / "BeamNG.drive.exe",
            bng_home / "Bin64" / "BeamNG.drive.x64.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("BeamNGBackend is not connected. Call connect() or reset() first.")

    def _read_config(self, config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, Mapping):
            return config.get(key, default)
        return getattr(config, key, default)

    def _build_placeholder_observation(self, scenario_config: Any, timestamp: float = 0.0) -> Observation:
        task = self._read_config(scenario_config, "task", None)
        start = self._read_task_value(task, "start", (0.0, 0.0))
        goal = self._read_task_value(task, "goal", (0.0, 0.0))
        vehicle_state = self._read_vehicle_state()
        if vehicle_state is None and self._last_observation is not None:
            vehicle_state = self._last_observation.vehicle_state
        if vehicle_state is None:
            vehicle_state = VehicleState(x=float(start[0]), y=float(start[1]))
        sensor_payload = self._sensor_cache or self._poll_sensors()
        front_rgb = self._sensor_value(sensor_payload, ("front_camera", "colour", "color", "annotation"))
        depth = self._sensor_value(sensor_payload, ("front_camera", "depth"))
        lidar = self._sensor_value(sensor_payload, ("roof_lidar", "points", "pointCloud", "pointcloud"))
        return Observation(
            timestamp=timestamp,
            vehicle_state=vehicle_state,
            goal=(float(goal[0]), float(goal[1])),
            front_rgb=front_rgb,
            depth=depth,
            lidar_points=lidar,
            info={
                "backend": "beamng",
                "status": "connected",
                "sensor_ids": sorted(self._sensors),
                "sensor_payload_keys": sorted(sensor_payload),
                "route": [list(point) for point in self._route],
                "note": "BeamNG pose and best-effort sensor payloads are read when available.",
            },
        )

    def _read_task_value(self, task: Any, key: str, default: Any) -> Any:
        if task is None:
            return default
        if isinstance(task, Mapping):
            return task.get(key, default)
        return getattr(task, key, default)

    def _beamng_level_for_config(self, config: Any) -> str:
        beamng_options = self._beamng_metadata(config)
        if beamng_options.get("level"):
            return str(beamng_options["level"])
        explicit_level = self._read_config(config, "beamng_level", None) or self._read_config(config, "level", None)
        if explicit_level:
            return str(explicit_level)
        if self._read_config(config, "backend", None) == "beamng":
            return str(self._read_config(config, "map", self.connection.level))
        return self.connection.level

    def _beamng_metadata(self, config: Any) -> dict[str, Any]:
        if isinstance(config, ScenarioConfig):
            return scenario_metadata_section(config, "beamng")
        metadata = self._read_config(config, "metadata", {})
        if isinstance(metadata, Mapping):
            value = metadata.get("beamng", {})
            return dict(value) if isinstance(value, Mapping) else {}
        return {}

    def _beamng_scenario_name_for_config(self, config: Any) -> str:
        beamng_options = self._beamng_metadata(config)
        base_name = str(beamng_options.get("scenario_name") or self._read_config(config, "scenario_id", self.connection.scenario_name))
        return f"{base_name}_{uuid.uuid4().hex[:8]}"

    def _beamng_vehicle_model_for_config(self, config: Any, vehicle_config: VehicleConfig | None) -> str:
        beamng_options = self._beamng_metadata(config)
        if beamng_options.get("vehicle_model"):
            return str(beamng_options["vehicle_model"])
        if vehicle_config is not None:
            return vehicle_config.template
        return self.connection.vehicle_model

    def _beamng_vehicle_start_for_config(self, config: Any) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        beamng_options = self._beamng_metadata(config)
        start = beamng_options.get("vehicle_start", {}) if isinstance(beamng_options.get("vehicle_start", {}), Mapping) else {}
        pos = self._tuple_float(start.get("pos"), (0.0, 0.0, 0.5), 3)
        rot_quat = self._tuple_float(start.get("rot_quat"), (0.0, 0.0, 0.0, 1.0), 4)
        return pos, rot_quat

    def _beamng_route_for_config(self, config: Any) -> list[tuple[float, float]]:
        beamng_options = self._beamng_metadata(config)
        raw_route = beamng_options.get("route", [])
        route: list[tuple[float, float]] = []
        if isinstance(raw_route, list):
            for point in raw_route:
                try:
                    route.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError, IndexError):
                    continue
        if route:
            return route
        task = self._read_config(config, "task", None)
        return [
            tuple(float(value) for value in self._read_task_value(task, "start", (0.0, 0.0))),
            tuple(float(value) for value in self._read_task_value(task, "goal", (0.0, 0.0))),
        ]

    def _tuple_float(self, value: Any, default: tuple[float, ...], length: int) -> tuple[float, ...]:
        if value is None:
            return default
        try:
            items = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return default
        return items if len(items) == length else default

    def _configure_visible_helpers(self) -> None:
        beamng_options = self._beamng_metadata(self._scenario_config)
        if not self._bng:
            return
        pos, _ = self._beamng_vehicle_start_for_config(self._scenario_config)
        self._configure_visible_camera()
        if bool(beamng_options.get("draw_route", False)) and self._route:
            debug = getattr(self._bng, "debug", None)
            add_spheres = getattr(debug, "add_spheres", None)
            if callable(add_spheres):
                points = [(x, y, pos[2] + 0.5) for x, y in self._route]
                try:
                    add_spheres(points, radii=[0.8] * len(points), colors=[(0.0, 1.0, 0.0, 0.8)] * len(points))
                except Exception:
                    pass

    def _configure_visible_camera(self, state: VehicleState | None = None) -> None:
        beamng_options = self._beamng_metadata(self._scenario_config)
        if str(beamng_options.get("camera_mode", "")).lower() not in {"orbit", "free", "follow"}:
            return
        if not self._bng:
            return
        camera = getattr(self._bng, "camera", None)
        set_player_mode = getattr(camera, "set_player_mode", None)
        if callable(set_player_mode):
            try:
                set_player_mode(
                    self.connection.vehicle_id,
                    "orbit",
                    {"distance": 12.0, "fov": 65.0, "rotation": (0.0, 0.0, 0.0)},
                )
                return
            except Exception:
                pass
        set_free = getattr(camera, "set_free", None)
        if not callable(set_free):
            return

        if state is None:
            pos, _ = self._beamng_vehicle_start_for_config(self._scenario_config)
            if len(self._route) >= 2:
                target = self._route[min(1, len(self._route) - 1)]
                dx = target[0] - pos[0]
                dy = target[1] - pos[1]
                yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else 0.0
            else:
                yaw = 0.0
            z = pos[2]
            x = pos[0]
            y = pos[1]
        else:
            yaw = float(state.yaw)
            x = float(state.x)
            y = float(state.y)
            z = float(state.z)

        cam_pos = (
            x - math.cos(yaw) * 9.0,
            y - math.sin(yaw) * 9.0,
            z + 5.0,
        )
        direction = (
            math.cos(yaw),
            math.sin(yaw),
            -0.35,
        )
        try:
            set_free(pos=cam_pos, direction=direction)
        except Exception:
            pass

    def _read_vehicle_state(self) -> VehicleState | None:
        vehicle = self._vehicle
        if vehicle is None:
            return None

        update_vehicle = getattr(vehicle, "update_vehicle", None)
        if callable(update_vehicle):
            try:
                update_vehicle()
            except Exception:
                pass

        state = getattr(vehicle, "state", None)
        if not isinstance(state, Mapping):
            return None

        pos = state.get("pos")
        if pos is None:
            pos = state.get("position")
        if pos is None or len(pos) < 2:
            return None

        direction = state.get("dir")
        if direction is None:
            direction = state.get("direction")
        yaw = 0.0
        if direction is not None and len(direction) >= 2:
            yaw = math.atan2(float(direction[1]), float(direction[0]))

        velocity = state.get("vel")
        if velocity is None:
            velocity = state.get("velocity")
        speed = 0.0
        if velocity is not None and len(velocity) >= 2:
            speed = math.sqrt(sum(float(component) ** 2 for component in velocity[:3]))

        return VehicleState(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]) if len(pos) >= 3 else 0.0,
            yaw=yaw,
            speed=speed,
        )

    def _update_motion_metrics(self, state: VehicleState) -> None:
        current = (float(state.x), float(state.y), float(state.z))
        if self._last_position is not None:
            self._distance_traveled += math.sqrt(sum((current[index] - self._last_position[index]) ** 2 for index in range(3)))
        self._last_position = current
        damage = self._read_damage()
        if damage > self._last_damage:
            self._collision_count += 1
        self._last_damage = max(self._last_damage, damage)

    def _read_damage(self) -> float:
        state = getattr(self._vehicle, "state", None)
        if not isinstance(state, Mapping):
            return self._last_damage
        for key in ("damage", "damage_total"):
            if key in state:
                try:
                    return float(state[key])
                except (TypeError, ValueError):
                    return self._last_damage
        return self._last_damage

    def _make_sensor(self, sensors_module: Any, sensor: Any) -> Any | None:
        sensor_type = getattr(sensor, "sensor_type", "")
        xyz = tuple(float(value) for value in getattr(sensor, "mount_xyz", (0.0, 0.0, 0.0)))
        rpy = tuple(float(value) for value in getattr(sensor, "mount_rpy", (0.0, 0.0, 0.0)))
        try:
            if sensor_type == "camera" and hasattr(sensors_module, "Camera"):
                Camera = getattr(sensors_module, "Camera")
                return Camera(
                    pos=xyz,
                    rot=rpy,
                    resolution=(int(getattr(sensor, "width", 640)), int(getattr(sensor, "height", 480))),
                    fov=float(getattr(sensor, "fov_deg", 90.0)),
                )
            if sensor_type == "lidar" and hasattr(sensors_module, "Lidar"):
                Lidar = getattr(sensors_module, "Lidar")
                return Lidar(pos=xyz, rot=rpy)
            if sensor_type == "imu" and hasattr(sensors_module, "IMU"):
                return getattr(sensors_module, "IMU")(pos=xyz, rot=rpy)
            if sensor_type == "gps" and hasattr(sensors_module, "GPS"):
                return getattr(sensors_module, "GPS")(pos=xyz, rot=rpy)
        except TypeError:
            return None
        return None

    def _poll_sensors(self) -> dict[str, Any]:
        vehicle = self._vehicle
        if vehicle is None:
            return {}
        poll = getattr(vehicle, "poll_sensors", None)
        if callable(poll):
            try:
                data = poll()
                return dict(data) if isinstance(data, Mapping) else {}
            except Exception:
                return {}
        return {}

    def _sensor_value(self, payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
        for sensor_id, sensor_payload in payload.items():
            if keys[0] not in str(sensor_id):
                continue
            if isinstance(sensor_payload, Mapping):
                for key in keys[1:]:
                    if key in sensor_payload:
                        return sensor_payload[key]
        return None

    def _missing_runtime_message(self, status: BackendStatus) -> str:
        return (
            "BeamNGBackend is optional and is not ready on this machine. "
            f"{status.message} Install BeamNG.tech, install `beamngpy`, and set `BNG_HOME` "
            "to the BeamNG.tech installation directory before calling connect/reset."
        )
