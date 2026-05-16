from __future__ import annotations

from offroad_sim.agents import make_agent
from offroad_sim.agents.route_world_model import RouteWorldModelAgent
from offroad_sim.core import Observation, VehicleState


def observation_at(x: float, y: float, *, goal: tuple[float, float] = (10.0, 0.0)) -> Observation:
    return Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=x, y=y, yaw=0.0, speed=1.0),
        goal=goal,
        info={"terrain_risk": 0.0},
    )


def test_route_world_model_agent_advances_waypoints() -> None:
    agent = RouteWorldModelAgent(route=[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)], world_model_name="simple_kinematic")
    agent.reset({"route": [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]})

    action = agent.act(observation_at(0.0, 0.0, goal=(10.0, 0.0)))

    assert action.throttle > 0.0
    assert agent.diagnostics()["target_waypoint_index"] >= 1
    assert agent.diagnostics()["target_waypoint"] == [5.0, 0.0]


def test_route_world_model_agent_uses_planner_when_configured() -> None:
    agent = RouteWorldModelAgent(
        world_model_name="simple_kinematic",
        planner_name="world_model_cem",
        planner_config={"horizon": 3, "num_samples": 8, "iterations": 1},
    )

    action = agent.act(observation_at(0.0, 0.0, goal=(10.0, 0.0)))

    assert -1.0 <= action.steer <= 1.0
    assert agent.diagnostics()["planner"] == "world_model_cem"


def test_route_world_model_agent_can_read_route_from_observation_info() -> None:
    agent = RouteWorldModelAgent(world_model_name="simple_kinematic")
    obs = observation_at(0.0, 0.0)
    obs.info["route"] = [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)]

    action = agent.act(obs)

    assert action.throttle > 0.0
    assert agent.diagnostics()["route_length"] == 3


def test_route_world_model_agent_is_registered() -> None:
    assert isinstance(make_agent("route_world_model"), RouteWorldModelAgent)
