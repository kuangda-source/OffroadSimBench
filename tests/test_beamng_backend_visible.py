from __future__ import annotations

from pathlib import Path
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
            def add_spheres(*args, **kwargs):
                module.debug_spheres = (args, kwargs)
                count = len(args[0]) if args else 0
                ids = list(range(module.next_debug_id, module.next_debug_id + count))
                module.next_debug_id += count
                return ids

            def remove_spheres(ids):
                module.removed_spheres.extend(ids)

            def add_triangle(*args, **kwargs):
                module.debug_triangles.append((args, kwargs))
                triangle_id = module.next_triangle_id
                module.next_triangle_id += 1
                return triangle_id

            def remove_triangle(triangle_id):
                module.removed_triangles.append(triangle_id)

            def add_polyline(*args, **kwargs):
                module.debug_polylines.append((args, kwargs))
                line_id = module.next_polyline_id
                module.next_polyline_id += 1
                return line_id

            def remove_polyline(line_id):
                module.removed_polylines.append(line_id)

            self.debug = SimpleNamespace(
                add_spheres=add_spheres,
                remove_spheres=remove_spheres,
                add_triangle=add_triangle,
                remove_triangle=remove_triangle,
                add_polyline=add_polyline,
                remove_polyline=remove_polyline,
            )
            module.bng = self

        def open(self, extensions=None, launch: bool = True, **kwargs: Any) -> None:
            module.open_launch = launch
            module.open_extensions = list(extensions or [])
            module.open_kwargs = dict(kwargs)

        def queue_lua_command(self, chunk: str, response: bool = False) -> Any:
            module.lua_commands.append((chunk, response))
            if "consumePickJson" in chunk:
                return '{"available":true,"sequence":7,"x":12.5,"y":-34.25,"z":101.2,"distance":8.0}'
            if "setEnabled" in chunk:
                return '{"available":false}'
            return None

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
    module.next_debug_id = 1
    module.removed_spheres = []
    module.next_triangle_id = 100
    module.debug_triangles = []
    module.removed_triangles = []
    module.next_polyline_id = 200
    module.debug_polylines = []
    module.removed_polylines = []
    module.lua_commands = []

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


def test_beamng_backend_keeps_route_free_manual_scenarios_route_free(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)
    scenario = {
        "scenario_id": "route_free_manual",
        "backend": "beamng",
        "task": {"start": [0.0, 0.0], "goal": [10.0, 0.0]},
        "metadata": {
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {"pos": [0.0, 0.0, 100.0], "yaw": 0.0},
                "drive_mode": "manual",
                "draw_route": False,
                "evaluation_route_mode": "none",
            }
        },
    }

    observation = backend.reset(scenario)

    assert "route" not in observation.info
    assert backend.get_metrics()["route_waypoint_count"] == 0


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

    assert fake_beamngpy.vehicle.last_control["steering"] == -1.0
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
    assert fake_beamngpy.vehicle.last_control["steering"] == -0.2
    assert "gear" not in fake_beamngpy.vehicle.last_control
    assert "clutch" not in fake_beamngpy.vehicle.last_control
    assert fake_beamngpy.shift_mode == "arcade"


def test_beamng_backend_hold_vehicle_uses_service_and_parking_brakes(fake_beamngpy: SimpleNamespace) -> None:
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    scenario = load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml")
    scenario.metadata["beamng"]["drive_mode"] = "manual"
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=vehicle)
    backend.reset(scenario)

    backend.hold_vehicle()

    assert fake_beamngpy.vehicle.last_control["steering"] == 0.0
    assert fake_beamngpy.vehicle.last_control["throttle"] == 0.0
    assert fake_beamngpy.vehicle.last_control["brake"] == 1.0
    assert fake_beamngpy.vehicle.last_control["parkingbrake"] == 1.0
    assert fake_beamngpy.vehicle.last_control["gear"] == 0
    assert fake_beamngpy.step_calls[-1] == 6


def test_beamng_backend_uses_topdown_preview_camera(fake_beamngpy: SimpleNamespace) -> None:
    scenario = {
        "scenario_id": "beamng_topdown_preview",
        "backend": "beamng",
        "task": {"start": [10.0, 20.0], "goal": [30.0, 40.0], "success_radius_m": 1.0},
        "metadata": {
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {"pos": [10.0, 20.0, 3.0], "rot_quat": [0.0, 0.0, 0.0, 1.0]},
                "camera_mode": "topdown",
                "camera_height_m": 75.0,
                "route": [[10.0, 20.0], [30.0, 40.0]],
                "drive_mode": "manual",
            }
        },
    }
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False))

    backend.reset(scenario)

    assert fake_beamngpy.camera_request["pos"] == (10.0, 20.0, 78.0)
    assert fake_beamngpy.camera_request["direction"] == (0.0, 0.0, -1.0)
    assert not hasattr(fake_beamngpy, "player_camera_request")


def test_beamng_backend_uses_high_follow_camera(fake_beamngpy: SimpleNamespace) -> None:
    scenario = {
        "scenario_id": "beamng_follow_camera",
        "backend": "beamng",
        "task": {"start": [10.0, 20.0], "goal": [30.0, 20.0], "success_radius_m": 1.0},
        "metadata": {
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {"pos": [10.0, 20.0, 3.0], "rot_quat": [0.0, 0.0, 0.0, 1.0]},
                "camera_mode": "follow",
                "camera_distance_m": 14.0,
                "camera_height_m": 13.0,
                "camera_lookahead_m": 2.0,
                "camera_pitch_deg": 35.0,
                "route": [[10.0, 20.0], [30.0, 20.0]],
                "drive_mode": "manual",
            }
        },
    }
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False))

    backend.reset(scenario)

    assert fake_beamngpy.player_camera_request[0] == "ego"
    assert fake_beamngpy.player_camera_request[1] == "orbit"
    assert fake_beamngpy.player_camera_request[2]["distance"] == 14.0
    assert fake_beamngpy.player_camera_request[2]["rotation"] == (0.0, -35.0, 0.0)
    assert not hasattr(fake_beamngpy, "camera_request")


def test_beamng_backend_updates_navigation_preview_without_reloading(fake_beamngpy: SimpleNamespace) -> None:
    scenario = {
        "scenario_id": "beamng_preview_update",
        "backend": "beamng",
        "task": {"start": [0.0, 0.0], "goal": [10.0, 0.0], "success_radius_m": 1.0},
        "metadata": {
            "task": {
                "task_type": "navigation_region_v1",
                "start_pose": {"pos": [0.0, 0.0, 0.5]},
                "goal": {"pos": [10.0, 0.0]},
                "region": {"polygon": [[0.0, -5.0], [12.0, -5.0], [12.0, 5.0], [0.0, 5.0]]},
            },
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {"pos": [0.0, 0.0, 0.5], "rot_quat": [0.0, 0.0, 0.0, 1.0]},
                "camera_mode": "topdown",
                "camera_height_m": 50.0,
                "route": [[0.0, 0.0], [10.0, 0.0]],
                "drive_mode": "manual",
                "draw_route": True,
                "draw_task_markers": True,
            },
        },
    }
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False))
    backend.reset(scenario)
    made_before = getattr(fake_beamngpy, "made", False)
    scenario["metadata"]["beamng"]["route"] = [[0.0, 0.0], [5.0, 2.0], [10.0, 0.0]]

    backend.update_navigation_preview(scenario)

    assert made_before is True
    assert fake_beamngpy.removed_spheres
    assert backend.get_metrics()["route_waypoint_count"] == 3
    assert fake_beamngpy.scenario_name.startswith("beamng_preview_update_")


def test_beamng_backend_draws_region_mask_triangles(fake_beamngpy: SimpleNamespace) -> None:
    scenario = {
        "scenario_id": "beamng_region_mask",
        "backend": "beamng",
        "task": {"start": [0.0, 0.0], "goal": [10.0, 0.0], "success_radius_m": 1.0},
        "metadata": {
            "task": {
                "task_type": "navigation_region_v1",
                "start_pose": {"pos": [0.0, 0.0, 0.5]},
                "goal": {"pos": [10.0, 0.0]},
                "region": {"polygon": [[0.0, -5.0], [12.0, -5.0], [12.0, 5.0], [0.0, 5.0]]},
            },
            "beamng": {
                "level": "gridmap_v2",
                "vehicle_start": {"pos": [0.0, 0.0, 0.5], "rot_quat": [0.0, 0.0, 0.0, 1.0]},
                "camera_mode": "topdown",
                "route": [[0.0, 0.0], [10.0, 0.0]],
                "drive_mode": "manual",
                "draw_route": True,
                "draw_task_markers": True,
            },
        },
    }
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False))

    backend.reset(scenario)
    scenario["metadata"]["task"]["region"]["polygon"] = [[0.0, -6.0], [12.0, -6.0], [12.0, 6.0], [0.0, 6.0]]
    backend.update_navigation_preview(scenario)

    assert len(fake_beamngpy.debug_triangles) >= 4
    assert fake_beamngpy.debug_triangles[-1][0][1][3] >= 0.4
    assert fake_beamngpy.debug_polylines
    line_args, _line_kwargs = fake_beamngpy.debug_polylines[-1]
    assert line_args[0][0] == line_args[0][-1]
    assert line_args[1][3] == 1.0
    assert fake_beamngpy.removed_triangles
    assert fake_beamngpy.removed_polylines


def test_beamng_backend_loads_lua_point_picker_and_consumes_pick(fake_beamngpy: SimpleNamespace, tmp_path: Path) -> None:
    bng_home = tmp_path / "BeamNG.tech"
    (bng_home / "Bin64").mkdir(parents=True)
    (bng_home / "Bin64" / "BeamNG.tech.x64.exe").write_text("", encoding="utf-8")
    vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
    backend = BeamNGBackend(
        connection=BeamNGConnectionConfig(
            bng_home=str(bng_home),
            launch=False,
            enable_point_picker=True,
        ),
        vehicle_config=vehicle,
    )

    backend.reset(load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml"))
    pick = backend.consume_point_picker()

    installed = bng_home / "lua" / "ge" / "extensions" / "offroadSimBench" / "pointPicker.lua"
    assert installed.exists()
    installed_text = installed.read_text(encoding="utf-8")
    assert "cameraMouseRayCast" in installed_text
    assert "M.onGuiUpdate = onGuiUpdate" in installed_text
    assert "last_mouse_down" in installed_text
    assert "last_mouse_down_edge" in installed_text
    assert "consumeOrCaptureMouseJson" in installed_text
    assert "showOverlay = false" in installed_text
    assert fake_beamngpy.open_extensions == ["offroadSimBench/pointPicker"]
    assert pick["available"] is True
    assert pick["sequence"] == 7
    assert pick["x"] == 12.5
    assert any("consumeOrCaptureMouseJson" in command for command, _response in fake_beamngpy.lua_commands)
    assert any("consumePickJson" in command for command, _response in fake_beamngpy.lua_commands)


def test_beamng_backend_reports_current_vehicle_pose(fake_beamngpy: SimpleNamespace) -> None:
    scenario = {
        "scenario_id": "beamng_pose_pick",
        "backend": "beamng",
        "task": {"start": [1.0, 2.0], "goal": [8.0, 2.0], "success_radius_m": 1.0},
        "metadata": {
            "beamng": {
                "level": "johnson_valley",
                "vehicle_start": {"pos": [1.0, 2.0, 3.0], "rot_quat": [0.0, 0.0, 0.0, 1.0]},
                "drive_mode": "manual",
                "steps_per_action": 1,
            }
        },
    }
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False))
    backend.reset(scenario)
    fake_beamngpy.vehicle.state["pos"] = [4.0, 5.0, 6.0]
    fake_beamngpy.vehicle.state["dir"] = [0.0, 1.0, 0.0]

    pose = backend.get_current_vehicle_pose()

    assert pose["available"] is True
    assert pose["x"] == 4.0
    assert pose["y"] == 5.0
    assert pose["z"] == 6.0
    assert pose["yaw"] == pytest.approx(1.570796, abs=1e-5)
    assert pose["level"] == "johnson_valley"


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
