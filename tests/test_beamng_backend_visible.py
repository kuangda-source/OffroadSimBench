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

        def advance(self) -> None:
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
    assert fake_beamngpy.spawned_pos == (0.0, 0.0, 0.5)
    assert fake_beamngpy.player_camera_request[0] == "ego"
    assert fake_beamngpy.player_camera_request[1] == "orbit"
    assert fake_beamngpy.debug_spheres
    assert backend.get_metrics()["route_waypoint_count"] == 4
    assert backend.get_metrics()["level"] == "gridmap_v2"


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
    assert fake_beamngpy.step_calls == [6]


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
    assert fake_beamngpy.spawned_pos == (0.0, 0.0, 0.5)
