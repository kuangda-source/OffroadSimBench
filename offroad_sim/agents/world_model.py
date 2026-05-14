"""Agents that use a world model for short-horizon risk checks."""

from __future__ import annotations

from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.core.types import Action, Observation
from offroad_sim.world_models import BaseWorldModel, SimpleKinematicWorldModel


class WorldModelAgent(OffroadAgent):
    """Rule-based goal follower with a world-model slowdown layer."""

    def __init__(
        self,
        world_model: BaseWorldModel | None = None,
        horizon: int = 12,
        risk_threshold: float = 0.6,
    ) -> None:
        self.world_model = world_model or SimpleKinematicWorldModel()
        self.base_agent = RuleBasedGoalAgent()
        self.horizon = horizon
        self.risk_threshold = risk_threshold

    def reset(self, scenario_info: Any) -> None:
        self.base_agent.reset(scenario_info)
        if isinstance(scenario_info, dict):
            self.world_model.reset(scenario_info)
        else:
            self.world_model.reset({"scenario": scenario_info})

    def act(self, obs: Observation) -> Action:
        action = self.base_agent.act(obs)
        prediction = self.world_model.predict(obs, action, horizon=self.horizon)
        max_risk = float(prediction.metadata.get("max_risk", 0.0))
        mean_risk = float(prediction.metadata.get("mean_risk", 0.0))

        if max_risk >= self.risk_threshold:
            return Action(
                steer=action.steer,
                throttle=min(action.throttle, 0.22),
                brake=max(action.brake, 0.12),
            )
        if mean_risk >= self.risk_threshold * 0.85:
            return Action(
                steer=action.steer,
                throttle=min(action.throttle, 0.35),
                brake=action.brake,
            )
        return action
