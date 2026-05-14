"""Basic agents for local backend smoke tests and demos."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.core.types import Action, Observation


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class RandomAgent(OffroadAgent):
    """Simple stochastic baseline for backend smoke tests."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, scenario_info: Any) -> None:
        return None

    def act(self, obs: Observation) -> Action:
        brake = 0.0
        if self.rng.random() < 0.05:
            brake = float(self.rng.uniform(0.1, 0.5))
        return Action(
            steer=float(self.rng.uniform(-0.8, 0.8)),
            throttle=float(self.rng.uniform(0.15, 0.65)),
            brake=brake,
        )


class RuleBasedGoalAgent(OffroadAgent):
    """Head toward the goal with a lightweight terrain-risk slowdown."""

    def __init__(self, cruise_throttle: float = 0.65) -> None:
        self.cruise_throttle = cruise_throttle

    def reset(self, scenario_info: Any) -> None:
        return None

    def act(self, obs: Observation) -> Action:
        state = obs.vehicle_state
        goal_x, goal_y = obs.goal
        target_heading = math.atan2(goal_y - state.y, goal_x - state.x)
        heading_error = _wrap_angle(target_heading - state.yaw)

        steer = _clip(heading_error / 0.75, -1.0, 1.0)
        throttle = self.cruise_throttle
        brake = 0.0

        if abs(heading_error) > 0.8:
            throttle = 0.25
        elif abs(heading_error) > 0.4:
            throttle = 0.45

        terrain_risk = float(obs.info.get("terrain_risk", 0.0))
        if terrain_risk > 0.7:
            throttle = min(throttle, 0.2)
            brake = 0.1
        elif terrain_risk > 0.5:
            throttle = min(throttle, 0.35)

        if state.speed > 9.0:
            throttle = min(throttle, 0.25)

        return Action(steer=steer, throttle=throttle, brake=brake)


class StopAgent(OffroadAgent):
    """Agent that commands a full stop every step."""

    def reset(self, scenario_info: Any) -> None:
        return None

    def act(self, obs: Observation) -> Action:
        return Action(steer=0.0, throttle=0.0, brake=1.0)


class KeyboardAgent(OffroadAgent):
    """Placeholder for future interactive keyboard control."""

    def reset(self, scenario_info: Any) -> None:
        return None

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError(
            "KeyboardAgent is a placeholder. Interactive keyboard control will be added later."
        )


def make_agent(name: str, seed: int | None = None, **kwargs: Any) -> OffroadAgent:
    from offroad_sim.agents.registry import make_agent as registry_make_agent

    return registry_make_agent(name, seed=seed, **kwargs)
