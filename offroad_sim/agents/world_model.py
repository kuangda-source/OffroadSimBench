"""Agents that use a world model for short-horizon risk checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.core.types import Action, Observation
from offroad_sim.planning import ActionPlanner, make_planner
from offroad_sim.world_models import BaseWorldModel, make_world_model


class WorldModelAgent(OffroadAgent):
    """Rule-based goal follower with a world-model slowdown layer."""

    def __init__(
        self,
        world_model: BaseWorldModel | None = None,
        world_model_name: str = "simple_kinematic",
        world_model_path: str | Path | None = None,
        planner: ActionPlanner | None = None,
        planner_name: str | None = None,
        planner_config: dict[str, Any] | None = None,
        horizon: int = 12,
        risk_threshold: float = 0.6,
    ) -> None:
        self.world_model = world_model or make_world_model(world_model_name, path=world_model_path)
        self.world_model_name = getattr(self.world_model, "model_type", world_model_name)
        self.world_model_path = str(world_model_path) if world_model_path is not None else None
        planner_kwargs = dict(planner_config or {})
        if planner_name == "le_wm_cem" and world_model_path is not None:
            planner_kwargs.setdefault("checkpoint_path", world_model_path)
        self.planner = planner or (make_planner(planner_name, **planner_kwargs) if planner_name else None)
        self.planner_name = getattr(self.planner, "planner_type", planner_name)
        self.base_agent = RuleBasedGoalAgent()
        self.horizon = horizon
        self.risk_threshold = risk_threshold
        self.last_prediction_metadata: dict[str, Any] = {}

    def reset(self, scenario_info: Any) -> None:
        self.base_agent.reset(scenario_info)
        if isinstance(scenario_info, dict):
            self.world_model.reset(scenario_info)
        else:
            self.world_model.reset({"scenario": scenario_info})

    def act(self, obs: Observation) -> Action:
        action = self.base_agent.act(obs)
        if self.planner is not None:
            planning = self.planner.plan(obs, self.world_model, reference_action=action)
            self.last_prediction_metadata = {
                "world_model": self.world_model_name,
                "world_model_path": self.world_model_path,
                "planner": self.planner_name,
                "planning": planning.metadata,
                "best_cost": planning.best_cost,
                "final_state": {
                    "x": planning.predicted_states[-1].x,
                    "y": planning.predicted_states[-1].y,
                    "yaw": planning.predicted_states[-1].yaw,
                    "speed": planning.predicted_states[-1].speed,
                }
                if planning.predicted_states
                else None,
            }
            return planning.first_action

        prediction = self.world_model.predict(obs, action, horizon=self.horizon)
        self.last_prediction_metadata = {
            "world_model": self.world_model_name,
            "world_model_path": self.world_model_path,
            "planner": None,
            "horizon": self.horizon,
            "prediction": prediction.metadata,
            "final_state": {
                "x": prediction.final_state.x,
                "y": prediction.final_state.y,
                "yaw": prediction.final_state.yaw,
                "speed": prediction.final_state.speed,
            }
            if prediction.final_state is not None
            else None,
        }
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

    def diagnostics(self) -> dict[str, Any]:
        return dict(self.last_prediction_metadata)
