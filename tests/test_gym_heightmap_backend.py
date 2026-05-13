from __future__ import annotations

import math
from pathlib import Path

from offroad_sim.agents import RandomAgent, RuleBasedGoalAgent
from offroad_sim.backends import GymHeightmapBackend
from offroad_sim.core import Action
from offroad_sim.scenarios import load_scenario_config


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "configs" / "scenarios" / "forest_trail_001.yaml"


def test_gym_heightmap_backend_reset_returns_maps() -> None:
    scenario = load_scenario_config(SCENARIO_PATH)
    backend = GymHeightmapBackend(seed=11)

    obs = backend.reset(scenario)

    assert obs.vehicle_state.x == scenario.task.start[0]
    assert obs.vehicle_state.y == scenario.task.start[1]
    assert obs.goal == scenario.task.goal
    assert obs.local_bev.shape == (4, 25, 25)
    assert obs.terrain_map.shape == (4, 128, 128)
    assert 0.0 <= obs.info["terrain_risk"] <= 1.0


def test_gym_heightmap_backend_step_updates_state_and_metrics() -> None:
    scenario = load_scenario_config(SCENARIO_PATH)
    backend = GymHeightmapBackend(seed=12)
    obs = backend.reset(scenario)

    result = backend.step(Action(steer=0.0, throttle=0.5, brake=0.0))

    assert result.observation.timestamp > obs.timestamp
    assert result.reward > -5.0
    assert result.info["distance_to_goal"] < 200.0
    assert backend.get_metrics()["episode_length"] == 1
    assert backend.get_metrics()["path_length"] > 0.0


def test_rule_based_agent_reaches_goal_in_default_scenario() -> None:
    scenario = load_scenario_config(SCENARIO_PATH)
    backend = GymHeightmapBackend(seed=7)
    agent = RuleBasedGoalAgent()
    obs = backend.reset(scenario)

    result = None
    for _ in range(800):
        result = backend.step(agent.act(obs))
        obs = result.observation
        if result.done:
            break

    metrics = backend.get_metrics()
    assert result is not None
    assert result.done is True
    assert metrics["success"] is True
    assert metrics["collision_count"] == 0
    assert metrics["distance_to_goal"] <= scenario.task.success_radius_m


def test_random_agent_can_drive_without_interface_errors() -> None:
    scenario = load_scenario_config(SCENARIO_PATH)
    backend = GymHeightmapBackend(seed=3)
    agent = RandomAgent(seed=3)
    obs = backend.reset(scenario)

    for _ in range(5):
        result = backend.step(agent.act(obs))
        obs = result.observation

    metrics = backend.get_metrics()
    assert math.isfinite(metrics["total_reward"])
    assert metrics["episode_length"] == 5

