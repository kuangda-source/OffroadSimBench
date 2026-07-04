from __future__ import annotations

import math

import pytest

from offroad_sim.agents import KeyboardAgent, ModelMPCAgent, RandomAgent, RuleBasedGoalAgent, StopAgent, make_agent
from offroad_sim.agents.world_model_direct import _stabilize_action
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning.types import PlanningResult


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
    positions = [(2.0, 2.0), (12.0, 8.0), (24.0, 24.0)]
    for step, (x, y) in enumerate(positions):
        observation.timestamp = float(step)
        observation.vehicle_state = VehicleState(x=x, y=y, yaw=0.0, speed=0.5)
        agent.act(observation)
        diagnostics = agent.diagnostics()
        targets.append(tuple(diagnostics["target"]))
        assert diagnostics["target_source"] == "route"
        assert diagnostics["route_target_count"] == 3
        assert diagnostics["target_in_region"] is True

    assert targets == [(12.0, 8.0), (24.0, 24.0), (28.0, 28.0)]


def test_region_explorer_holds_route_target_until_reached() -> None:
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
        waypoint_radius_m=2.0,
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
        observation.vehicle_state = VehicleState(x=3.0 + step * 0.5, y=2.0, yaw=0.0, speed=0.5)
        agent.act(observation)
        targets.append(tuple(agent.diagnostics()["target"]))

    assert targets == [(12.0, 8.0), (12.0, 8.0), (12.0, 8.0)]


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


def test_world_model_direct_agent_uses_local_subgoal_inside_region() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4}, local_subgoal_distance_m=12.0)
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=1.0),
        goal=(35.0, 2.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]]},
                "goal": {"pos": [35.0, 2.0], "radius": 4.0},
            }
        },
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.throttle >= 0.0
    assert diagnostics["target_goal"] == [35.0, 2.0]
    assert diagnostics["local_subgoal"] == [14.0, 2.0]
    assert diagnostics["planner_goal"] == [14.0, 2.0]
    assert diagnostics["route_used"] is False


def test_world_model_direct_agent_offsets_local_subgoal_after_stuck() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 1, "num_samples": 4, "seed": 4}, local_subgoal_distance_m=12.0)
    agent._stuck_steps = 24
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=0.0),
        goal=(35.0, 2.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, -10.0], [40.0, -10.0], [40.0, 20.0], [0.0, 20.0]]},
                "goal": {"pos": [35.0, 2.0], "radius": 4.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert subgoal[0] > 10.0
    assert abs(subgoal[1] - 2.0) >= 3.0
    assert 0.0 <= subgoal[0] <= 40.0
    assert -10.0 <= subgoal[1] <= 20.0


def test_world_model_direct_agent_can_escape_laterally_when_direct_goal_side_is_blocked() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 1, "num_samples": 4, "seed": 4}, local_subgoal_distance_m=12.0)
    agent._stuck_steps = 36
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=1317.08, y=-107.46, yaw=-2.57, speed=0.0),
        goal=(1245.995, -189.268),
        info={
            "navigation_region": {
                "region": {
                    "polygon": [[1173.822, -138.863], [1217.423, -222.645], [1376.998, -124.37], [1311.841, -37.816]]
                },
                "goal": {"pos": [1245.995, -189.268], "radius": 12.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert subgoal[0] < observation.vehicle_state.x
    assert subgoal[1] > observation.vehicle_state.y


def test_world_model_direct_agent_keeps_stall_memory_through_low_speed_jitter() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 1, "num_samples": 4, "seed": 4}, local_subgoal_distance_m=22.0)

    class SlowPlanner:
        def plan(self, observation, world_model, *, reference_action=None, score_actions=None):
            return PlanningResult(
                actions=[Action(steer=0.08, throttle=0.78, brake=0.0)],
                predicted_states=[],
                costs=[0.0],
                best_cost=0.0,
                metadata={"planner": "slow_test"},
            )

    agent.planner = SlowPlanner()
    info = {
        "navigation_region": {
            "region": {
                "polygon": [[1173.822, -138.863], [1217.423, -222.645], [1376.998, -124.37], [1311.841, -37.816]]
            },
            "goal": {"pos": [1245.995, -189.268], "radius": 12.0},
        }
    }

    action = Action()
    for step in range(44):
        jitter = 0.04 if step % 2 else 0.0
        observation = Observation(
            timestamp=float(step),
            vehicle_state=VehicleState(x=1318.88 + jitter, y=-107.10 - jitter * 0.5, yaw=-2.24, speed=0.02),
            goal=(1245.995, -189.268),
            info=info,
        )
        action = agent.act(observation)

    diagnostics = agent.diagnostics()
    local_subgoal = diagnostics["local_subgoal"]
    dx = observation.goal[0] - observation.vehicle_state.x
    dy = observation.goal[1] - observation.vehicle_state.y
    distance = math.hypot(dx, dy)
    direct_subgoal = (
        observation.vehicle_state.x + dx / distance * agent.local_subgoal_distance_m,
        observation.vehicle_state.y + dy / distance * agent.local_subgoal_distance_m,
    )
    assert diagnostics["stuck_recovery"] is True
    assert action.gear is None
    assert math.hypot(local_subgoal[0] - direct_subgoal[0], local_subgoal[1] - direct_subgoal[1]) > 8.0
    assert 1173.0 <= local_subgoal[0] <= 1377.5
    assert -223.0 <= local_subgoal[1] <= -37.0


def test_world_model_direct_agent_uses_turn_arc_subgoal_for_large_heading_error() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 1, "num_samples": 4, "seed": 4}, local_subgoal_distance_m=12.0)
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=1329.177, y=-109.397, yaw=2.001, speed=0.1),
        goal=(1245.995, -189.268),
        info={
            "navigation_region": {
                "region": {
                    "polygon": [[1173.822, -138.863], [1217.423, -222.645], [1376.998, -124.37], [1311.841, -37.816]]
                },
                "goal": {"pos": [1245.995, -189.268], "radius": 12.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert subgoal[0] < observation.vehicle_state.x
    assert subgoal[1] > observation.vehicle_state.y


def test_world_model_direct_agent_can_use_experience_corridor_for_local_subgoal() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
    )
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=1.0),
        goal=(35.0, 2.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [40.0, 0.0], [40.0, 25.0], [0.0, 25.0]]},
                "goal": {"pos": [35.0, 2.0], "radius": 4.0},
                "experience_route": [[2.0, 2.0], [2.0, 14.0], [20.0, 14.0], [35.0, 2.0]],
            }
        },
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.throttle >= 0.0
    assert diagnostics["target_goal"] == [35.0, 2.0]
    assert diagnostics["local_subgoal"] == [2.0, 14.0]
    assert diagnostics["planner_goal"] == [2.0, 14.0]
    assert diagnostics["route_used"] is False
    assert diagnostics["experience_corridor_used"] is True


def test_world_model_direct_agent_can_use_model_support_points_for_local_subgoal() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
        use_model_support_subgoals=True,
    )
    agent.world_model.metadata = {
        "support_points": [[2.0, 2.0], [2.0, 14.0], [20.0, 14.0], [35.0, 2.0]],
    }
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=1.0),
        goal=(35.0, 2.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [40.0, 0.0], [40.0, 25.0], [0.0, 25.0]]},
                "goal": {"pos": [35.0, 2.0], "radius": 4.0},
            }
        },
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.throttle >= 0.0
    assert diagnostics["target_goal"] == [35.0, 2.0]
    assert diagnostics["local_subgoal"] == [2.0, 14.0]
    assert diagnostics["planner_goal"] == [2.0, 14.0]
    assert diagnostics["route_used"] is False
    assert diagnostics["experience_corridor_used"] is False
    assert diagnostics["model_support_subgoal_used"] is True


def test_world_model_direct_agent_keeps_model_support_routes_segmented() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
        use_model_support_subgoals=True,
    )
    agent.world_model.metadata = {
        "support_points": [[0.0, 0.0], [0.0, 10.0], [100.0, 0.0], [100.0, 10.0]],
        "support_routes": [
            [[0.0, 0.0], [0.0, 10.0]],
            [[100.0, 0.0], [100.0, 10.0]],
        ],
    }
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=9.0, yaw=0.0, speed=1.0),
        goal=(100.0, 10.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[-5.0, -5.0], [105.0, -5.0], [105.0, 15.0], [-5.0, 15.0]]},
                "goal": {"pos": [100.0, 10.0], "radius": 4.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert subgoal == (0.0, 10.0)
    assert agent._model_support_subgoal_used is True


def test_world_model_direct_agent_bridges_connected_model_support_routes() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
        use_model_support_subgoals=True,
    )
    agent.world_model.metadata = {
        "support_routes": [
            [[0.0, 0.0], [0.0, 10.0]],
            [[8.0, 12.0], [20.0, 12.0]],
        ],
    }
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=9.0, yaw=0.0, speed=1.0),
        goal=(20.0, 12.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[-5.0, -5.0], [25.0, -5.0], [25.0, 20.0], [-5.0, 20.0]]},
                "goal": {"pos": [20.0, 12.0], "radius": 4.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert subgoal is not None
    assert subgoal[0] > 0.0
    assert math.hypot(subgoal[0] - 20.0, subgoal[1] - 12.0) < math.hypot(0.0 - 20.0, 10.0 - 12.0)
    assert agent._model_support_subgoal_used is True


def test_world_model_direct_agent_can_use_unordered_support_field_for_local_subgoal() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
        use_model_support_field_subgoals=True,
    )
    agent.world_model.metadata = {
        "support_points": [[35.0, 2.0], [20.0, 14.0], [12.0, 8.0], [2.0, 2.0]],
    }
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=2.0, y=2.0, yaw=0.0, speed=1.0),
        goal=(35.0, 2.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[0.0, 0.0], [40.0, 0.0], [40.0, 25.0], [0.0, 25.0]]},
                "goal": {"pos": [35.0, 2.0], "radius": 4.0},
            }
        },
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action.throttle >= 0.0
    assert diagnostics["target_goal"] == [35.0, 2.0]
    assert diagnostics["local_subgoal"] == [12.0, 8.0]
    assert diagnostics["planner_goal"] == [12.0, 8.0]
    assert diagnostics["route_used"] is False
    assert diagnostics["experience_corridor_used"] is False
    assert diagnostics["model_support_subgoal_used"] is False
    assert diagnostics["model_support_field_subgoal_used"] is True


def test_world_model_direct_agent_support_field_does_not_bridge_far_segments() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
        use_model_support_field_subgoals=True,
    )
    agent.world_model.metadata = {
        "support_routes": [
            [[0.0, 0.0], [5.0, 10.0]],
            [[100.0, 0.0], [100.0, 10.0]],
        ],
    }
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(100.0, 0.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[-5.0, -5.0], [105.0, -5.0], [105.0, 15.0], [-5.0, 15.0]]},
                "goal": {"pos": [100.0, 0.0], "radius": 4.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert subgoal[0] < 20.0
    assert subgoal[1] > 4.0
    assert agent._model_support_subgoal_used is False
    assert agent._model_support_field_subgoal_used is True


def test_world_model_direct_agent_support_field_ignores_nearby_past_points() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
        use_model_support_field_subgoals=True,
    )
    agent.world_model.metadata = {
        "support_points": [[0.0, 0.0], [0.8, 0.0], [1.4, 0.1], [8.0, 10.0], [20.0, 10.0], [35.0, 0.0]],
    }
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=1.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(35.0, 0.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[-5.0, -5.0], [40.0, -5.0], [40.0, 20.0], [-5.0, 20.0]]},
                "goal": {"pos": [35.0, 0.0], "radius": 4.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert math.hypot(subgoal[0] - observation.vehicle_state.x, subgoal[1] - observation.vehicle_state.y) >= 4.0
    assert subgoal[1] > 4.0
    assert agent._model_support_field_subgoal_used is True


def test_world_model_direct_agent_support_field_requires_goal_progress() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
        use_model_support_field_subgoals=True,
    )
    agent.world_model.metadata = {
        "support_points": [[0.0, 10.0], [20.0, 10.0]],
    }
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(35.0, 0.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[-5.0, -5.0], [40.0, -5.0], [40.0, 20.0], [-5.0, 20.0]]},
                "goal": {"pos": [35.0, 0.0], "radius": 4.0},
            }
        },
    )

    subgoal = agent._local_subgoal(observation)

    assert subgoal[0] > 5.0
    assert math.hypot(subgoal[0] - 35.0, subgoal[1]) < 35.0
    assert agent._model_support_field_subgoal_used is True


def test_world_model_direct_agent_uses_local_subgoal_progress_for_stuck_detection() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        local_subgoal_distance_m=12.0,
    )

    class ForwardPlanner:
        def plan(self, observation, world_model, *, reference_action=None, score_actions=None):
            return PlanningResult(
                actions=[Action(steer=0.0, throttle=0.55, brake=0.0)],
                predicted_states=[],
                costs=[0.0],
                best_cost=0.0,
                metadata={"planner": "forward_test"},
            )

    agent.planner = ForwardPlanner()
    info = {
        "navigation_region": {
            "region": {"polygon": [[-5.0, -5.0], [30.0, -5.0], [30.0, 20.0], [-5.0, 20.0]]},
            "experience_route": [[0.0, 0.0], [0.0, 12.0], [20.0, 12.0], [20.0, 0.0]],
        }
    }

    for step in range(24):
        obs = Observation(
            timestamp=float(step),
            vehicle_state=VehicleState(x=0.0, y=float(step) * 0.4, yaw=1.57, speed=0.05),
            goal=(20.0, 0.0),
            info=info,
        )
        action = agent.act(obs)

    diagnostics = agent.diagnostics()
    assert action.gear is None
    assert diagnostics["experience_corridor_used"] is True
    assert diagnostics["stuck_recovery"] is False


def test_world_model_direct_agent_detects_physical_stall_when_subgoal_changes() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})

    for step in range(24):
        observation = Observation(
            timestamp=float(step),
            vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
            goal=(12.0, float(step % 2) * 2.0),
            info={},
        )
        action, stuck = agent._progress_filter(
            Action(steer=0.05, throttle=0.55, brake=0.0),
            Action(steer=0.08, throttle=0.45, brake=0.0),
            observation,
        )

    assert stuck is True
    assert action.gear is None


def test_world_model_direct_agent_interpolates_long_experience_corridor_segments() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 1, "num_samples": 4, "seed": 4},
        local_subgoal_distance_m=22.0,
    )
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.0),
        goal=(60.0, 0.0),
        info={
            "navigation_region": {
                "region": {"polygon": [[-5.0, -5.0], [70.0, -5.0], [70.0, 5.0], [-5.0, 5.0]]},
                "goal": {"pos": [60.0, 0.0], "radius": 4.0},
                "experience_route": [[0.0, 0.0], [4.0, 0.0], [60.0, 0.0]],
            }
        },
    )

    agent.act(observation)

    diagnostics = agent.diagnostics()
    assert diagnostics["local_subgoal"] == [22.0, 0.0]
    assert diagnostics["planner_goal"] == [22.0, 0.0]
    assert diagnostics["experience_corridor_used"] is True


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

    assert -0.4 <= action.steer <= -0.25
    assert action.throttle >= 0.75
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


def test_world_model_direct_agent_straight_low_speed_recovery_releases_brake() -> None:
    agent = make_agent("world_model_direct", planner_config={"horizon": 4, "num_samples": 16, "seed": 4})
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
        goal=(20.0, 0.0),
        info={},
    )

    for _ in range(24):
        action, stuck = agent._progress_filter(
            Action(steer=0.05, throttle=0.45, brake=0.0),
            Action(steer=0.08, throttle=0.45, brake=0.0),
            observation,
        )

    assert stuck is True
    assert action.gear is None
    assert action.throttle >= 0.32
    assert action.brake == 0.0
    assert abs(action.steer) <= 0.25


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


def test_world_model_direct_agent_uses_reverse_only_after_prolonged_stuck_when_enabled() -> None:
    agent = make_agent(
        "world_model_direct",
        planner_config={"horizon": 4, "num_samples": 16, "seed": 4},
        allow_reverse_recovery=True,
        reverse_recovery_after_steps=48,
    )
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.02),
        goal=(0.0, -20.0),
        info={},
    )

    for _ in range(24):
        early_action, early_stuck = agent._progress_filter(
            Action(steer=-0.75, throttle=0.4, brake=0.0),
            Action(steer=-0.9, throttle=0.45, brake=0.0),
            observation,
        )

    assert early_stuck is True
    assert early_action.gear is None

    for _ in range(31):
        late_action, late_stuck = agent._progress_filter(
            Action(steer=-0.75, throttle=0.4, brake=0.0),
            Action(steer=-0.9, throttle=0.45, brake=0.0),
            observation,
        )

    assert late_stuck is True
    assert late_action.gear == -1
    assert 0.4 <= late_action.throttle <= 0.65
    assert abs(late_action.steer) <= 0.1


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
