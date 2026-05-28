from __future__ import annotations

from pathlib import Path

from offroad_sim.scenarios import ScenarioConfig, load_scenario_config
from offroad_sim.tasks import load_navigation_region_task
from offroad_sim.vehicles import (
    CameraConfig,
    LidarConfig,
    VehicleConfig,
    load_vehicle_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_vehicle_config_loads_from_yaml() -> None:
    config = load_vehicle_config(ROOT / "configs" / "vehicles" / "ugv_medium.yaml")

    assert isinstance(config, VehicleConfig)
    assert config.vehicle_id == "ugv_medium_001"
    assert config.template == "medium_offroad"
    assert config.mass_kg == 1200.0
    assert config.max_steer_deg == 35.0
    assert isinstance(config.sensors[0], CameraConfig)
    assert isinstance(config.sensors[1], LidarConfig)


def test_scenario_config_loads_from_yaml() -> None:
    config = load_scenario_config(ROOT / "configs" / "scenarios" / "forest_trail_001.yaml")

    assert isinstance(config, ScenarioConfig)
    assert config.scenario_id == "forest_trail_001"
    assert config.backend == "gym_heightmap"
    assert config.terrain.type == "forest"
    assert config.task.max_time_sec == 180.0
    assert config.task.goal == (80.0, 60.0)
    assert config.metrics.terrain_risk is True


def test_johnson_valley_demo_uses_stable_manual_control_timing() -> None:
    task = load_navigation_region_task(ROOT / "configs" / "tasks" / "beamng_johnson_valley_nav_test.yaml")

    assert task.max_steps >= 900
    assert int(task.beamng["steps_per_action"]) <= 8
    assert float(task.beamng["ai_line_speed"]) <= 8.0
