from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from offroad_sim.backends import BeamNGBackend, BeamNGConnectionConfig
from offroad_sim.backends.registry import BackendStatus
from offroad_sim.core import Action
from offroad_sim.evaluation import run_episode
from offroad_sim.scenarios import load_scenario_config
from offroad_sim.vehicles import load_vehicle_config


@pytest.fixture
def fake_beamngpy(monkeypatch) -> SimpleNamespace:
    module = SimpleNamespace()

    class FakeBeamNGpy:
        def __init__(self, host: str, port: int, **kwargs: Any) -> None:
            self.host = host
            self.port = port
            self.kwargs = kwargs
            self.scenario = SimpleNamespace(load=lambda scenario: setattr(module, "loaded_scenario", scenario), start=lambda: setattr(module, "started", True))
            self.camera = SimpleNamespace(
                set_free=lambda **kwargs: setattr(module, "camera_request", kwargs),
                set_player_mode=lambda vehicle, mode, config: setattr(module, "player_camera_request", (vehicle, mode, config)),
            )
            self.debug = SimpleNamespace(add_spheres=lambda *args, **kwargs: setattr(module, "debug_spheres", (args, kwargs)))
            module.bng = self

        def open(self, launch: bool = True) -> None:
            module.open_launch = launch

        def step(self, steps: int) -> None:
            module.step_calls.append(steps)
            if getattr(module, "vehicle", None) is not None:
                module.vehicle.advance()

        def close(self) -> None:
            module.closed = True

    class FakeScenario:
        def __init__(self, level: str, name: str) -> None:
            self.level = level
            self.name = name
            module.scenario_level = level
            module.scenario_name = name

        def add_vehicle(self, vehicle: Any, *, pos: tuple[float, float, float], rot_quat: tuple[float, float, float, float]) -> None:
            module.vehicle = vehicle
            module.spawned_vehicle_model = vehicle.model
            module.spawned_pos = pos
            module.spawned_rot_quat = rot_quat
            vehicle.state["pos"] = list(pos)

        def make(self, bng: Any) -> None:
            module.made = True

    class FakeVehicle:
        def __init__(self, vehicle_id: str, *, model: str, **kwargs: Any) -> None:
            self.vehicle_id = vehicle_id
            self.model = model
            self.kwargs = kwargs
            self.state = {"pos": [0.0, 0.0, 0.5], "dir": [1.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "damage": 0.0}
            self.last_control = {}
            self.sensors: dict[str, Any] = {}

        def control(self, **kwargs: Any) -> None:
            self.last_control = kwargs

        def set_shift_mode(self, mode: str) -> None:
            module.shift_mode = mode

        def ai_set_line(self, line: list[dict[str, Any]], cling: bool = True) -> None:
            module.ai_line = line
            module.ai_line_cling = cling

        def advance(self) -> None:
            if getattr(module, "ai_line", None):
                target = module.ai_line[min(1, len(module.ai_line) - 1)]["pos"]
                dx = float(target[0]) - float(self.state["pos"][0])
                dy = float(target[1]) - float(self.state["pos"][1])
                distance = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                speed = min(2.0, distance)
                self.state["vel"] = [dx / distance * speed, dy / distance * speed, 0.0]
                self.state["pos"] = [
                    float(self.state["pos"][0]) + self.state["vel"][0],
                    float(self.state["pos"][1]) + self.state["vel"][1],
                    float(self.state["pos"][2]),
                ]
                return
            throttle = float(self.last_control.get("throttle", 0.0))
            brake = float(self.last_control.get("brake", 0.0))
            speed = max(0.0, float(self.state["vel"][0]) + throttle - brake)
            self.state["vel"] = [speed, 0.0, 0.0]
            self.state["pos"] = [float(self.state["pos"][0]) + speed, float(self.state["pos"][1]), 0.5]

        def update_vehicle(self) -> None:
            return None

        def attach_sensor(self, sensor_id: str, sensor: Any) -> None:
            self.sensors[sensor_id] = sensor

        def poll_sensors(self) -> dict[str, Any]:
            return {}

    class FakeCamera:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeLidar:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    module.BeamNGpy = FakeBeamNGpy
    module.Scenario = FakeScenario
    module.Vehicle = FakeVehicle
    module.sensors = SimpleNamespace(Camera=FakeCamera, Lidar=FakeLidar)
    module.step_calls = []

    def fake_import(name: str) -> Any:
        if name == "beamngpy":
            return module
        if name == "beamngpy.sensors":
            return module.sensors
        raise ImportError(name)

    monkeypatch.setattr(
        BeamNGBackend,
        "runtime_status",
        classmethod(lambda cls, bng_home=None: BackendStatus("beamng", True, "fake ready", {})),
    )
    monkeypatch.setattr("offroad_sim.backends.beamng_backend.importlib.import_module", fake_import)
    return module


def test_beamng_backend_uses_visible_scenario_metadata(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)

    observation = backend.reset(load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml"))

    assert observation.info["backend"] == "beamng"
    assert fake_beamngpy.bng.kwargs["gfx"] == "vk"
    assert fake_beamngpy.scenario_level == "gridmap_v2"
    assert fake_beamngpy.scenario_name.startswith("beamng_visible_autodrive_")
    assert fake_beamngpy.scenario_name != "beamng_visible_autodrive"
    assert fake_beamngpy.spawned_vehicle_model == "pickup"
    assert fake_beamngpy.spawned_pos == (1.37432313, -167.098877, 100.6)
    assert fake_beamngpy.player_camera_request[0] == "ego"
    assert fake_beamngpy.player_camera_request[1] == "orbit"
    assert fake_beamngpy.debug_spheres
    assert backend.get_metrics()["route_waypoint_count"] == 4
    assert backend.get_metrics()["level"] == "gridmap_v2"
    assert backend.get_metrics()["drive_mode"] == "ai_line"
    assert fake_beamngpy.ai_line_cling is True
    assert fake_beamngpy.ai_line[0]["speed"] == 12.0


def test_beamng_backend_reports_motion_damage_and_sensors(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)
    backend.reset(load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml"))

    result = backend.step(Action(throttle=0.4))
    metrics = backend.get_metrics()

    assert result.info["step_count"] == 1
    assert metrics["episode_length"] == 1
    assert metrics["distance_traveled"] > 0.0
    assert metrics["collision_count"] == 0
    assert metrics["sensor_count"] == 2
    assert metrics["horizontal_distance_traveled"] > 0.0
    assert fake_beamngpy.step_calls == [18]


def test_beamng_backend_clamps_control_actions(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    scenario = load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml")
    scenario.metadata["beamng"]["drive_mode"] = "manual"
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)
    backend.reset(scenario)

    backend.step(Action(steer=2.0, throttle=-0.5, brake=1.5))

    assert fake_beamngpy.vehicle.last_control["steering"] == 1.0
    assert fake_beamngpy.vehicle.last_control["throttle"] == 0.0
    assert fake_beamngpy.vehicle.last_control["brake"] == 1.0


def test_beamng_backend_sends_manual_agent_control_as_adas(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    scenario = load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml")
    scenario.metadata["beamng"]["drive_mode"] = "manual"
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)
    backend.reset(scenario)

    backend.step(Action(steer=0.2, throttle=0.6, brake=0.0))

    assert fake_beamngpy.vehicle.last_control["is_adas"] is True
    assert "gear" not in fake_beamngpy.vehicle.last_control
    assert "clutch" not in fake_beamngpy.vehicle.last_control
    assert fake_beamngpy.shift_mode == "arcade"


def test_beamng_backend_fallback_observation_uses_spawn_yaw(fake_beamngpy: SimpleNamespace, monkeypatch) -> None:
    scenario = {
        "scenario_id": "beamng_spawn_yaw",
        "backend": "beamng",
        "task": {"start": [1.0, 2.0], "goal": [8.0, 2.0], "success_radius_m": 1.0},
        "metadata": {
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {
                    "pos": [1.0, 2.0, 3.0],
                    "yaw": -0.5,
                    "rot_quat": [0.0, 0.0, -0.247404, 0.968912],
                },
                "drive_mode": "manual",
                "steps_per_action": 1,
            }
        },
    }
    monkeypatch.setattr(BeamNGBackend, "_read_vehicle_state", lambda self: None)
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False))

    observation = backend.reset(scenario)

    assert observation.vehicle_state.x == 1.0
    assert observation.vehicle_state.y == 2.0
    assert observation.vehicle_state.z == 3.0
    assert observation.vehicle_state.yaw == -0.5


def test_beamng_backend_skips_manual_control_for_ai_line(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)
    backend.reset(load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml"))

    backend.step(Action(steer=1.0, throttle=1.0, brake=1.0))

    assert fake_beamngpy.vehicle.last_control == {}


def test_beamng_backend_terminates_when_goal_radius_is_reached(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    scenario = {
        "scenario_id": "beamng_goal_stop",
        "backend": "beamng",
        "task": {"start": [0.0, 0.0], "goal": [2.0, 0.0], "success_radius_m": 0.5},
        "metadata": {
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {"pos": [0.0, 0.0, 0.5], "rot_quat": [0.0, 0.0, 0.0, 1.0]},
                "route": [[0.0, 0.0], [2.0, 0.0]],
                "drive_mode": "ai_line",
                "steps_per_action": 1,
            }
        },
    }
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)
    backend.reset(scenario)

    result = backend.step(Action())

    assert result.terminated is True
    assert result.done is True
    assert result.info["goal_reached"] is True
    assert result.info["distance_to_goal"] <= 0.5
    assert backend.get_metrics()["goal_reached"] is True


def test_beamng_observation_exposes_navigation_region_task(fake_beamngpy: SimpleNamespace) -> None:
    scenario = {
        "scenario_id": "beamng_region_info",
        "backend": "beamng",
        "task": {"start": [0.0, 0.0], "goal": [2.0, 0.0], "success_radius_m": 0.5},
        "metadata": {
            "task": {
                "task_type": "navigation_region_v1",
                "region": {"polygon": [[0.0, -2.0], [10.0, -2.0], [10.0, 2.0], [0.0, 2.0]]},
            },
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {"pos": [0.0, 0.0, 0.5], "rot_quat": [0.0, 0.0, 0.0, 1.0]},
                "drive_mode": "manual",
            },
        },
    }
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False))

    observation = backend.reset(scenario)

    assert observation.info["navigation_region"]["task_type"] == "navigation_region_v1"
    assert observation.info["navigation_region"]["region"]["polygon"][0] == [0.0, -2.0]


def test_run_episode_passes_vehicle_config_to_beamng(fake_beamngpy: SimpleNamespace) -> None:
    result = run_episode(
        backend_name="beamng",
        scenario="configs/scenarios/beamng_visible_autodrive.yaml",
        agent_name="stop",
        vehicle="configs/vehicles/ugv_medium.yaml",
        max_steps=1,
    )

    assert result.backend == "beamng"
    assert fake_beamngpy.spawned_vehicle_model == "pickup"
    assert fake_beamngpy.spawned_pos == (1.37432313, -167.098877, 100.6)
