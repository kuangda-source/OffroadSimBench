from __future__ import annotations

from offroad_sim.agents import make_agent
from offroad_sim.agents.route_world_model import RouteWorldModelAgent
from offroad_sim.core import Action, Observation, VehicleState


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


def test_route_world_model_agent_advances_past_missed_waypoint() -> None:
    agent = RouteWorldModelAgent(
        route=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
        waypoint_radius_m=3.0,
        world_model_name="simple_kinematic",
    )
    agent.reset({"route": [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]})

    agent.act(observation_at(15.0, 4.0, goal=(20.0, 0.0)))

    assert agent.diagnostics()["target_waypoint_index"] == 2
    assert agent.diagnostics()["target_waypoint"] == [20.0, 0.0]


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


def test_route_world_model_agent_keeps_visible_demo_moving_when_planner_stalls() -> None:
    class StallingPlanner:
        def reset(self, scenario_info) -> None:
            return None

        def act(self, obs: Observation) -> Action:
            return Action(steer=0.2, throttle=0.0, brake=1.0)

        def diagnostics(self) -> dict[str, str]:
            return {"planner": "fake_stalling"}

        def close(self) -> None:
            return None

    agent = RouteWorldModelAgent(route=[(0.0, 0.0), (20.0, 0.0)], world_model_name="simple_kinematic")
    agent.inner = StallingPlanner()

    action = agent.act(observation_at(0.0, 0.0))

    assert action.throttle >= 0.35
    assert action.brake <= 0.15
    assert agent.diagnostics()["progress_guard"] is True


def test_route_world_model_agent_releases_brake_when_starting_from_rest() -> None:
    class DraggingPlanner:
        def reset(self, scenario_info) -> None:
            return None

        def act(self, obs: Observation) -> Action:
            return Action(steer=-0.8, throttle=0.35, brake=0.18)

        def diagnostics(self) -> dict[str, str]:
            return {"planner": "fake_dragging"}

        def close(self) -> None:
            return None

    agent = RouteWorldModelAgent(route=[(0.0, 0.0), (20.0, 0.0)], world_model_name="simple_kinematic")
    agent.inner = DraggingPlanner()

    action = agent.act(Observation(timestamp=0.0, vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.05), goal=(20.0, 0.0)))

    assert action.throttle >= 0.55
    assert action.brake == 0.0
    assert agent.diagnostics()["progress_guard"] is True


def test_route_world_model_agent_releases_brake_below_crawl_speed() -> None:
    class DraggingPlanner:
        def reset(self, scenario_info) -> None:
            return None

        def act(self, obs: Observation) -> Action:
            return Action(steer=0.9, throttle=0.45, brake=0.15)

        def diagnostics(self) -> dict[str, str]:
            return {"planner": "fake_dragging"}

        def close(self) -> None:
            return None

    agent = RouteWorldModelAgent(route=[(0.0, 0.0), (20.0, 0.0)], world_model_name="simple_kinematic")
    agent.inner = DraggingPlanner()

    action = agent.act(Observation(timestamp=0.0, vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.5), goal=(20.0, 0.0)))

    assert action.throttle >= 0.55
    assert action.brake == 0.0
    assert agent.diagnostics()["progress_guard"] is True


def test_route_world_model_agent_uses_reference_steer_when_low_speed_planner_conflicts() -> None:
    class ConflictingPlanner:
        def reset(self, scenario_info) -> None:
            return None

        def act(self, obs: Observation) -> Action:
            return Action(steer=-1.0, throttle=0.7, brake=0.0)

        def diagnostics(self) -> dict[str, str]:
            return {"planner": "fake_conflicting"}

        def close(self) -> None:
            return None

    agent = RouteWorldModelAgent(route=[(0.0, 0.0), (20.0, 0.0)], world_model_name="simple_kinematic")
    agent.inner = ConflictingPlanner()

    action = agent.act(Observation(timestamp=0.0, vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.5), goal=(20.0, 0.0)))

    assert abs(action.steer) < 0.05
    assert action.throttle >= 0.55
    assert action.brake == 0.0
    assert agent.diagnostics()["progress_guard"] is True
    assert agent.diagnostics()["execution_controller"] == "model_guided_route_tracker"
    assert agent.diagnostics()["planner_action"]["steer"] == -1.0
    assert agent.diagnostics()["executed_action"]["steer"] == action.steer


def test_route_world_model_agent_caps_low_speed_reference_steer() -> None:
    class BrakingPlanner:
        def reset(self, scenario_info) -> None:
            return None

        def act(self, obs: Observation) -> Action:
            return Action(steer=0.1, throttle=-0.2, brake=0.4)

        def diagnostics(self) -> dict[str, str]:
            return {"planner": "fake_braking"}

        def close(self) -> None:
            return None

    agent = RouteWorldModelAgent(route=[(0.0, 0.0), (0.0, 20.0)], world_model_name="simple_kinematic")
    agent.inner = BrakingPlanner()

    action = agent.act(Observation(timestamp=0.0, vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.25), goal=(0.0, 20.0)))

    assert action.steer == 0.35
    assert action.throttle >= 0.55
    assert action.brake == 0.0
    assert agent.diagnostics()["progress_guard"] is True


def test_route_world_model_agent_boosts_throttle_after_stuck_steps() -> None:
    class MildPlanner:
        def reset(self, scenario_info) -> None:
            return None

        def act(self, obs: Observation) -> Action:
            return Action(steer=0.0, throttle=0.2, brake=0.0)

        def diagnostics(self) -> dict[str, str]:
            return {"planner": "fake_mild"}

        def close(self) -> None:
            return None

    agent = RouteWorldModelAgent(route=[(0.0, 0.0), (20.0, 0.0)], world_model_name="simple_kinematic")
    agent.inner = MildPlanner()
    obs = Observation(timestamp=0.0, vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.05), goal=(20.0, 0.0))

    action = Action()
    for _ in range(18):
        action = agent.act(obs)

    assert action.throttle >= 0.9
    assert abs(action.steer) <= 0.2
    assert action.brake == 0.0
    assert agent.diagnostics()["stuck_recovery"] is True


def test_route_world_model_agent_is_registered() -> None:
    assert isinstance(make_agent("route_world_model"), RouteWorldModelAgent)
