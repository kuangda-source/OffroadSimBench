from __future__ import annotations

import pytest

from offroad_sim.agents import KeyboardAgent, RandomAgent, RuleBasedGoalAgent, StopAgent, make_agent
from offroad_sim.core import Observation, VehicleState


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

