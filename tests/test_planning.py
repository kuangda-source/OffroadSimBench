from __future__ import annotations

import pytest

from offroad_sim.agents import WorldModelAgent
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning import StableWorldModelUnavailableError, WorldModelCEMPlanner, default_planner_registry
from offroad_sim.planning.stablewm import LeWMCEMPlanner
from offroad_sim.world_models import SimpleKinematicWorldModel


def _observation() -> Observation:
    return Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(8.0, 0.0),
    )


def test_planner_registry_reports_local_and_lewm_planners() -> None:
    registry = default_planner_registry()

    assert {"world_model_cem", "le_wm_cem"}.issubset(set(registry.names()))
    assert registry.status("world_model_cem").available is True


def test_world_model_cem_plans_toward_goal() -> None:
    planner = WorldModelCEMPlanner(horizon=4, num_samples=16, iterations=2, seed=4)
    result = planner.plan(_observation(), SimpleKinematicWorldModel(), reference_action=Action(throttle=0.4))

    assert len(result.actions) == 4
    assert result.first_action.throttle >= 0.0
    assert result.predicted_states[-1].x > 0.0
    assert result.metadata["planner"] == "world_model_cem"


def test_world_model_agent_can_use_planner() -> None:
    agent = WorldModelAgent(planner_name="world_model_cem", planner_config={"horizon": 3, "num_samples": 12, "iterations": 1})

    action = agent.act(_observation())
    diagnostics = agent.diagnostics()

    assert isinstance(action, Action)
    assert diagnostics["planner"] == "world_model_cem"
    assert diagnostics["best_cost"] >= 0.0


def test_lewm_planner_requires_checkpoint() -> None:
    planner = LeWMCEMPlanner()

    with pytest.raises(StableWorldModelUnavailableError, match="checkpoint"):
        planner.plan(_observation(), SimpleKinematicWorldModel())


def test_lewm_planner_passes_navigation_region_to_cost_model() -> None:
    torch = pytest.importorskip("torch")
    planner = LeWMCEMPlanner()
    observation = _observation()
    observation.info["navigation_region"] = {
        "region": {"polygon": [[0.0, -2.0], [10.0, -2.0], [10.0, 2.0], [0.0, 2.0]]}
    }

    info = planner._observation_to_info(observation, torch)

    assert "region_polygon" in info
    assert tuple(info["region_polygon"].shape) == (1, 1, 4, 2)
