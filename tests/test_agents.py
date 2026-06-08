from __future__ import annotations

import pytest

from offroad_sim.agents import KeyboardAgent, ModelMPCAgent, RandomAgent, RuleBasedGoalAgent, StopAgent, make_agent
from offroad_sim.agents.world_model_direct import _stabilize_action
from offroad_sim.core import Action, Observation, VehicleState


def make_observation() -> Observation:
    return Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(10.0, 0.0),
        info={"terrain_risk": 0.0},
    )


def test_random_agent_action_is_normalized() -> None:
    action = RandomAgent(seed=1).act(make_observation())

    assert -1.0 <= action.steer <= 1.0
    assert 0.0 <= action.throttle <= 1.0
    assert 0.0 <= action.brake <= 1.0


def test_stop_agent_commands_brake() -> None:
    action = StopAgent().act(make_observation())

    assert action.throttle == 0.0
    assert action.brake == 1.0


def test_rule_based_agent_steers_toward_goal() -> None:
    action = RuleBasedGoalAgent().act(make_observation())

    assert abs(action.steer) < 0.05
    assert action.throttle > 0.0


def test_keyboard_agent_is_placeholder() -> None:
    with pytest.raises(NotImplementedError):
        KeyboardAgent().act(make_observation())


def test_make_agent_factory() -> None:
    assert isinstance(make_agent("random"), RandomAgent)
    assert isinstance(make_agent("stop"), StopAgent)
    assert isinstance(make_agent("rule-based"), RuleBasedGoalAgent)
    assert isinstance(make_agent("keyboard"), KeyboardAgent)


def test_make_agent_supports_model_mpc() -> None:
    agent = make_agent("model_mpc", planner_config={"horizon": 8, "num_samples": 24, "iterations": 2})

    assert isinstance(agent, ModelMPCAgent)


def test_region_explorer_targets_points_inside_navigation_region() -> None:
    agent = make_agent("region_explorer", seed=3)
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=0.5),
        goal=(20.0, 20.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]}
            }
        },
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert -1.0 <= action.steer <= 1.0
    assert 0.0 <= action.throttle <= 1.0
    assert diagnostics["agent"] == "region_explorer"
    assert diagnostics["target_in_region"] is True


def test_world_model_direct_agent_ignores_expert_route_info() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(20.0, 0.0),
        info={"route": [[0.0, 0.0], [0.0, 20.0]]},
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.throttle >= 0.0
    assert diagnostics["agent"] == "world_model_direct"
    assert diagnostics["target_goal"] == [20.0, 0.0]
    assert diagnostics["route_used"] is False


def test_world_model_direct_agent_keeps_low_speed_progress() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.4),
        goal=(20.0, 0.0),
        info={},
    )

    action = _stabilize_action(
        Action(steer=0.95, throttle=0.15, brake=0.0),
        Action(steer=0.2, throttle=0.5, brake=0.0),
        observation,
    )

    assert action.throttle >= 0.78
    assert action.brake == 0.0
    assert abs(action.steer) <= 0.25


def test_world_model_direct_agent_recovers_from_low_speed_no_progress() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
        goal=(20.0, 0.0),
        info={},
    )

    for _ in range(24):
        action, stuck = agent._progress_filter(Action(steer=0.6, throttle=0.7, brake=0.0), Action(steer=0.6, throttle=0.5, brake=0.0), observation)

    assert stuck is True
    assert action.throttle == 1.0
    assert action.brake == 0.0
    assert abs(action.steer) <= 0.18


def test_model_mpc_agent_uses_route_and_mpc_diagnostics() -> None:
    agent = ModelMPCAgent(planner_config={"horizon": 8, "num_samples": 24, "seed": 3})
    observation = make_observation()
    observation.info["route"] = [(0.0, 0.0), (0.0, 12.0)]
    observation.goal = (0.0, 12.0)

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.steer > 0.1
    assert action.throttle > 0.0
    assert diagnostics["agent"] == "model_mpc"
    assert diagnostics["planner"] == "navigation_mpc"
    assert diagnostics["target_waypoint"] == [0.0, 12.0]


def test_model_mpc_agent_brakes_inside_navigation_goal_radius() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (10.0, 0.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=9.0, y=0.0, yaw=0.0, speed=4.0),
        goal=(10.0, 0.0),
        info={"navigation_region": {"goal": {"pos": [10.0, 0.0], "radius": 2.0}}},
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.throttle == 0.0
    assert action.brake == 1.0
    assert diagnostics["terminal_stop"] is True


def test_model_mpc_agent_recovers_from_low_speed_stuck_turn() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (0.0, 20.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.7, speed=0.05),
        goal=(0.0, 20.0),
        info={},
    )

    for _ in range(14):
        action = agent._execution_filter(Action(steer=-1.0, throttle=0.45), Action(steer=-1.0, throttle=0.25), observation)

    assert action.throttle >= 0.8
    assert action.steer <= -0.85
    assert agent.diagnostics().get("stuck_recovery") is True


def test_model_mpc_agent_recovers_from_low_speed_no_progress() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (20.0, 0.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
        goal=(20.0, 0.0),
        info={},
    )

    for _ in range(14):
        action = agent._execution_filter(Action(steer=0.0, throttle=0.55), Action(steer=0.0, throttle=0.55), observation)

    assert action.throttle == 1.0
    assert action.brake == 0.0
    assert agent.diagnostics().get("stuck_recovery") is True


def test_model_mpc_agent_uses_route_lookahead_for_dense_waypoints() -> None:
    agent = ModelMPCAgent(
        route=[(0.0, 0.0), (-8.0, 19.0), (-24.0, 25.0), (-36.0, 10.0)],
        route_lookahead_m=24.0,
        planner_config={"horizon": 4, "num_samples": 12, "seed": 2},
    )
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=2.7, speed=1.0),
        goal=(-36.0, 10.0),
        info={},
    )

    target = agent._target_for(observation)

    assert target == (-24.0, 25.0)
    assert agent.cursor == 2


def test_model_mpc_agent_slows_before_high_speed_sharp_turn() -> None:
    agent = ModelMPCAgent(planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=7.5),
        goal=(20.0, 0.0),
        info={},
    )

    action = agent._execution_filter(Action(steer=-0.9, throttle=0.55), Action(steer=-0.9, throttle=0.45), observation)

    assert abs(action.steer) <= 0.45
    assert action.throttle == 0.0
    assert action.brake >= 0.15


def test_model_mpc_agent_limits_sweeping_turn_speed() -> None:
    agent = ModelMPCAgent(planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=5.8),
        goal=(20.0, 0.0),
        info={},
    )

    action = agent._execution_filter(Action(steer=0.6, throttle=0.65), Action(steer=0.6, throttle=0.45), observation)

    assert abs(action.steer) <= 0.55
    assert action.throttle <= 0.15
    assert action.brake >= 0.08
