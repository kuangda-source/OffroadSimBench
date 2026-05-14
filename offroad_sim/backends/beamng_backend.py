"""Optional BeamNG.tech backend adapter.

The module is import-safe on machines without BeamNG.tech or beamngpy. Real
connection work is delayed until ``connect``/``reset``.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from offroad_sim.backends.base import OffroadSimBackend
from offroad_sim.backends.registry import BackendStatus
from offroad_sim.core import Action, Observation, StepResult, VehicleState
from offroad_sim.scenarios import ScenarioConfig
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
        self._last_observation: Observation | None = None
        self._scenario_config: Any = None
        self._step_count = 0
        self._connected = False
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
        self._bng = BeamNGpy(self.connection.host, self.connection.port, **kwargs)
        self._bng.open(launch=self.connection.launch)
        self._connected = True

    def load_scenario(self, scenario_config: ScenarioConfig | Mapping[str, Any] | None = None) -> None:
        self._ensure_connected()
        beamngpy = importlib.import_module("beamngpy")
        Scenario = getattr(beamngpy, "Scenario")
        level = self._beamng_level_for_config(scenario_config)
        scenario_name = self._read_config(scenario_config, "scenario_id", self.connection.scenario_name)
        self._scenario = Scenario(level, scenario_name)

    def spawn_vehicle(self, vehicle_config: VehicleConfig | None = None) -> None:
        self._ensure_connected()
        beamngpy = importlib.import_module("beamngpy")
        Vehicle = getattr(beamngpy, "Vehicle")
        config = vehicle_config or self.vehicle_config
        model = config.template if config is not None else self.connection.vehicle_model
        try:
            self._vehicle = Vehicle(self.connection.vehicle_id, model=model, license="OSB")
        except TypeError:
            self._vehicle = Vehicle(self.connection.vehicle_id, model=model, licence="OSB")
        if self._scenario is not None:
            self._scenario.add_vehicle(self._vehicle, pos=(0.0, 0.0, 0.0), rot_quat=(0.0, 0.0, 0.0, 1.0))

    def attach_sensors(self, vehicle_config: VehicleConfig | None = None) -> None:
        self._ensure_connected()
        self.vehicle_config = vehicle_config or self.vehicle_config
        # Real camera/lidar/imu mapping will live here once beamngpy is installed.

    def reset(self, scenario_config: Any = None) -> Observation:
        self._scenario_config = scenario_config
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

        self._step_count = 0
        self._last_observation = self._build_placeholder_observation(scenario_config)
        return self._last_observation

    def step(self, action: Action) -> StepResult:
        self._ensure_connected()
        if self._vehicle is not None and hasattr(self._vehicle, "control"):
            self._vehicle.control(throttle=action.throttle, steering=action.steer, brake=action.brake)
        if self._bng is not None and hasattr(self._bng, "step"):
            self._bng.step(self.connection.steps_per_action)

        self._step_count += 1
        self._last_observation = self._build_placeholder_observation(
            self._scenario_config,
            timestamp=float(self._step_count),
        )
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
            "level": self.connection.level,
        }

    def close(self) -> None:
        if self._bng is not None and hasattr(self._bng, "close"):
            self._bng.close()
        self._bng = None
        self._scenario = None
        self._vehicle = None
        self._last_observation = None
        self._scenario_config = None
        self._connected = False
        self._step_count = 0

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
        return Observation(
            timestamp=timestamp,
            vehicle_state=vehicle_state,
            goal=(float(goal[0]), float(goal[1])),
            info={
                "backend": "beamng",
                "status": "connected",
                "note": "BeamNG pose is read when available; camera/lidar array mapping is reserved for the sensor pass.",
            },
        )

    def _read_task_value(self, task: Any, key: str, default: Any) -> Any:
        if task is None:
            return default
        if isinstance(task, Mapping):
            return task.get(key, default)
        return getattr(task, key, default)

    def _beamng_level_for_config(self, config: Any) -> str:
        explicit_level = self._read_config(config, "beamng_level", None) or self._read_config(config, "level", None)
        if explicit_level:
            return str(explicit_level)
        if self._read_config(config, "backend", None) == "beamng":
            return str(self._read_config(config, "map", self.connection.level))
        return self.connection.level

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

    def _missing_runtime_message(self, status: BackendStatus) -> str:
        return (
            "BeamNGBackend is optional and is not ready on this machine. "
            f"{status.message} Install BeamNG.tech, install `beamngpy`, and set `BNG_HOME` "
            "to the BeamNG.tech installation directory before calling connect/reset."
        )
