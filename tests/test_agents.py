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


def test_region_explorer_can_bias_collection_toward_navigation_goal() -> None:
    agent = make_agent("region_explorer", seed=3, goal_bias_interval=1)
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

    assert action.throttle > 0.0
    assert diagnostics["target"] == [20.0, 20.0]
    assert diagnostics["target_source"] == "goal"
    assert diagnostics["target_in_region"] is True


def test_region_explorer_samples_goal_corridor_targets_between_vehicle_and_goal() -> None:
    agent = make_agent("region_explorer", seed=3, goal_bias_interval=0, goal_corridor_interval=1)
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=0.5),
        goal=(22.0, 2.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [30.0, 0.0], [30.0, 12.0], [0.0, 12.0]]}
            }
        },
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.throttle > 0.0
    assert diagnostics["target_source"] == "goal_corridor"
    assert diagnostics["target_in_region"] is True
    assert 2.0 < diagnostics["target"][0] < 22.0
    assert abs(diagnostics["target"][1] - 2.0) < 3.0


def test_region_explorer_can_cycle_coverage_targets_inside_region() -> None:
    agent = make_agent(
        "region_explorer",
        seed=3,
        goal_bias_interval=0,
        goal_corridor_interval=0,
        coverage_grid_size=3,
        coverage_target_interval=1,
        max_target_steps=1,
    )
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=0.5),
        goal=(28.0, 28.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]}
            }
        },
    )

    targets: list[tuple[float, float]] = []
    for step in range(4):
        observation.timestamp = float(step)
        agent.act(observation)
        diagnostics = agent.diagnostics()
        targets.append(tuple(diagnostics["target"]))
        assert diagnostics["target_source"] == "coverage"
        assert diagnostics["target_in_region"] is True
        assert diagnostics["coverage_target_count"] >= 4

    assert len(set(targets)) == len(targets)


def test_region_explorer_shuffles_coverage_targets_by_seed() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=0.5),
        goal=(28.0, 28.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]}
            }
        },
    )

    def first_targets(seed: int) -> list[tuple[float, float]]:
        agent = make_agent(
            "region_explorer",
            seed=seed,
            goal_bias_interval=0,
            goal_corridor_interval=0,
            coverage_grid_size=4,
            coverage_target_interval=1,
            max_target_steps=1,
        )
        targets: list[tuple[float, float]] = []
        for step in range(6):
            observation.timestamp = float(step)
            agent.act(observation)
            targets.append(tuple(agent.diagnostics()["target"]))
        return targets

    assert first_targets(3) != first_targets(4)


def test_region_explorer_can_follow_route_aware_curriculum_targets() -> None:
    agent = make_agent(
        "region_explorer",
        seed=3,
        goal_bias_interval=0,
        goal_corridor_interval=0,
        coverage_grid_size=0,
        coverage_target_interval=0,
        route_target_interval=1,
        route_lateral_m=0.0,
        max_target_steps=1,
    )
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=0.5),
        goal=(28.0, 28.0),
        info={
            "route": [[2.0, 2.0], [12.0, 8.0], [24.0, 24.0], [28.0, 28.0]],
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]}
            },
        },
    )

    targets: list[tuple[float, float]] = []
    for step in range(3):
        observation.timestamp = float(step)
        agent.act(observation)
        diagnostics = agent.diagnostics()
        targets.append(tuple(diagnostics["target"]))
        assert diagnostics["target_source"] == "route"
        assert diagnostics["route_target_count"] == 3
        assert diagnostics["target_in_region"] is True

    assert targets == [(12.0, 8.0), (24.0, 24.0), (28.0, 28.0)]


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


def test_world_model_direct_agent_brakes_inside_navigation_goal_radius() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=19.5, y=0.0, yaw=0.0, speed=2.0),
        goal=(20.0, 0.0),
        info={"navigation_region": {"goal_radius": 2.0}},
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.steer == 0.0
    assert action.throttle == 0.0
    assert action.brake == 1.0
    assert action.gear == 0
    assert diagnostics["goal_stop"] is True


def test_world_model_direct_agent_leaves_forward_gear_to_backend_shift_mode() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.5),
        goal=(20.0, 0.0),
        info={},
    )

    action = _stabilize_action(
        Action(steer=0.0, throttle=0.5, brake=0.0),
        Action(steer=0.0, throttle=0.5, brake=0.0),
        observation,
    )

    assert action.gear is None


def test_world_model_direct_agent_latches_goal_hold_after_overshoot() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})
    first = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=19.5, y=0.0, yaw=0.0, speed=3.0),
        goal=(20.0, 0.0),
        info={"navigation_region": {"goal_radius": 2.0}},
    )
    overshot = Observation(
        timestamp=0.1,
        vehicle_state=VehicleState(x=22.4, y=0.0, yaw=0.0, speed=2.0),
        goal=(20.0, 0.0),
        info={"navigation_region": {"goal_radius": 2.0}},
    )

    agent.act(first)
    action = agent.act(overshot)
    diagnostics = agent.diagnostics()

    assert action.throttle == 0.0
    assert action.brake == 1.0
    assert action.gear == 0
    assert diagnostics["goal_hold_latched"] is True


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


def test_world_model_direct_agent_uses_controlled_sharp_turn_from_low_speed() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.2),
        goal=(0.0, -20.0),
        info={},
    )

    action = _stabilize_action(
        Action(steer=0.0, throttle=0.1, brake=0.0),
        Action(steer=-1.0, throttle=0.25, brake=0.0),
        observation,
    )

    assert action.steer <= -0.7
    assert 0.35 <= action.throttle <= 0.5
    assert action.brake == 0.0


def test_world_model_direct_agent_limits_speed_for_medium_speed_sharp_turn() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=5.8),
        goal=(0.0, -20.0),
        info={},
    )

    action = _stabilize_action(
        Action(steer=-0.9, throttle=0.65, brake=0.0),
        Action(steer=-0.9, throttle=0.25, brake=0.0),
        observation,
    )

    assert abs(action.steer) <= 0.55
    assert action.throttle <= 0.2
    assert action.brake >= 0.08


def test_world_model_direct_agent_rejects_medium_speed_steer_against_goal_bearing() -> None:
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=4.1),
        goal=(-20.0, -20.0),
        info={},
    )

    action = _stabilize_action(
        Action(steer=0.75, throttle=0.45, brake=0.0),
        Action(steer=-1.0, throttle=0.25, brake=0.0),
        observation,
    )

    assert action.steer < 0.0
    assert abs(action.steer) <= 0.75
    assert action.throttle <= 0.2
    assert action.brake >= 0.08


def test_world_model_direct_agent_recovers_from_low_speed_no_progress() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
        goal=(0.0, -20.0),
        info={},
    )

    for _ in range(24):
        action, stuck = agent._progress_filter(Action(steer=-0.75, throttle=0.4, brake=0.0), Action(steer=-0.9, throttle=0.45, brake=0.0), observation)

    assert stuck is True
    assert action.gear is None
    assert 0.2 <= action.throttle <= 0.55
    assert action.brake <= 0.25
    assert abs(action.steer) <= 0.9


def test_world_model_direct_agent_turns_again_after_straight_recovery_burst() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
        goal=(0.0, -20.0),
        info={},
    )

    for _ in range(36):
        action, stuck = agent._progress_filter(Action(steer=-0.75, throttle=0.4, brake=0.0), Action(steer=-0.9, throttle=0.45, brake=0.0), observation)

    assert stuck is True
    assert action.gear is None
    assert 0.4 <= action.throttle <= 0.65
    assert action.brake == 0.0
    assert action.steer <= -0.65


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
    assert action.gear == 0
    assert diagnostics["terminal_stop"] is True


def test_model_mpc_agent_leaves_forward_gear_to_backend_shift_mode() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (20.0, 0.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(20.0, 0.0),
        info={},
    )

    action = agent._execution_filter(
        Action(steer=0.0, throttle=0.55, brake=0.0),
        Action(steer=0.0, throttle=0.55, brake=0.0),
        observation,
    )

    assert action.gear is None


def test_model_mpc_agent_latches_goal_hold_after_overshoot() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (10.0, 0.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    first = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=9.0, y=0.0, yaw=0.0, speed=4.0),
        goal=(10.0, 0.0),
        info={"navigation_region": {"goal": {"pos": [10.0, 0.0], "radius": 2.0}}},
    )
    overshot = Observation(
        timestamp=0.1,
        vehicle_state=VehicleState(x=12.5, y=0.0, yaw=0.0, speed=2.5),
        goal=(10.0, 0.0),
        info={"navigation_region": {"goal": {"pos": [10.0, 0.0], "radius": 2.0}}},
    )

    agent.act(first)
    action = agent.act(overshot)
    diagnostics = agent.diagnostics()

    assert action.throttle == 0.0
    assert action.brake == 1.0
    assert action.gear == 0
    assert diagnostics["goal_hold_latched"] is True


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

    assert action.gear == -1
    assert 0.4 <= action.throttle <= 0.65
    assert abs(action.steer) <= 0.1
    assert agent.diagnostics().get("stuck_recovery") is True


def test_model_mpc_agent_turns_after_straight_recovery_burst() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (0.0, 20.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.7, speed=0.05),
        goal=(0.0, 20.0),
        info={},
    )

    for _ in range(26):
        action = agent._execution_filter(Action(steer=-1.0, throttle=0.45), Action(steer=-1.0, throttle=0.25), observation)

    assert action.gear is None
    assert 0.45 <= action.throttle <= 0.75
    assert action.steer <= -0.65
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

    assert action.gear == -1
    assert 0.4 <= action.throttle <= 0.65
    assert action.brake == 0.0
    assert agent.diagnostics().get("stuck_recovery") is True


def test_model_mpc_agent_does_not_recover_when_low_speed_progresses_to_target() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (20.0, 0.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})

    for step in range(18):
        observation = Observation(
            timestamp=step * 0.1,
            vehicle_state=VehicleState(x=step * 0.04, y=0.0, yaw=0.0, speed=0.08),
            goal=(20.0, 0.0),
            info={},
        )
        action = agent._execution_filter(Action(steer=0.0, throttle=0.55), Action(steer=0.0, throttle=0.55), observation)

    assert action.gear is None
    assert agent.diagnostics().get("stuck_recovery") is False
    assert agent.diagnostics().get("stuck_steps") < 12


def test_model_mpc_agent_resets_recovery_after_target_progress_resumes() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (20.0, 0.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    stationary = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
        goal=(20.0, 0.0),
        info={},
    )
    for _ in range(14):
        agent._execution_filter(Action(steer=0.0, throttle=0.55), Action(steer=0.0, throttle=0.55), stationary)
    assert agent.diagnostics().get("stuck_recovery") is True

    for step in range(8):
        moving = Observation(
            timestamp=1.0 + step * 0.1,
            vehicle_state=VehicleState(x=1.0 + step * 0.2, y=0.0, yaw=0.0, speed=0.2),
            goal=(20.0, 0.0),
            info={},
        )
        action = agent._execution_filter(Action(steer=0.0, throttle=0.55), Action(steer=0.0, throttle=0.55), moving)

    assert action.gear is None
    assert agent.diagnostics().get("stuck_recovery") is False


def test_model_mpc_agent_follows_reference_steer_when_low_speed_plan_conflicts() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (0.0, 20.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.4),
        goal=(0.0, 20.0),
        info={},
    )

    action = agent._execution_filter(Action(steer=-0.9, throttle=0.75), Action(steer=0.9, throttle=0.45), observation)

    assert action.steer > 0.4
    assert action.throttle <= 0.65
    assert action.gear is None


def test_model_mpc_agent_suppresses_low_speed_oversteer_when_reference_is_aligned() -> None:
    agent = ModelMPCAgent(route=[(0.0, 0.0), (20.0, 0.0)], planner_config={"horizon": 4, "num_samples": 12, "seed": 2})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.08),
        goal=(20.0, 0.0),
        info={},
    )

    action = agent._execution_filter(Action(steer=-0.8, throttle=0.65), Action(steer=0.04, throttle=0.55), observation)

    assert abs(action.steer) <= 0.25
    assert action.throttle >= 0.6
    assert action.brake == 0.0


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


def test_model_mpc_agent_keeps_local_target_before_sharp_route_corner() -> None:
    agent = ModelMPCAgent(
        route=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (10.0, 20.0)],
        route_lookahead_m=24.0,
        planner_config={"horizon": 4, "num_samples": 12, "seed": 2},
    )
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(10.0, 20.0),
        info={},
    )

    target = agent._target_for(observation)

    assert target == (10.0, 0.0)
    assert agent.cursor == 1


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
