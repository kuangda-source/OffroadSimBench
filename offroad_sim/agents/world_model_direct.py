"""Route-free world-model MPC agent."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.agents.model_mpc import _action_dict
from offroad_sim.core import Action, Observation
from offroad_sim.planning import make_planner
from offroad_sim.planning.navigation_mpc import NavigationMPCPlanner
from offroad_sim.world_models import make_world_model


class WorldModelDirectAgent(OffroadAgent):
    """Drive toward the task goal with a world-model planner and no expert route."""

    def __init__(
        self,
        world_model_name: str = "simple_kinematic",
        world_model_path: str | Path | None = None,
        planner_name: str | None = None,
        planner_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.world_model_name = str(world_model_name or "simple_kinematic")
        self.world_model_path = str(world_model_path) if world_model_path else None
        self.world_model = make_world_model(self.world_model_name, path=world_model_path) if world_model_path else make_world_model(self.world_model_name)
        self.reference_agent = RuleBasedGoalAgent(cruise_throttle=0.5)
        self.planner_name = str(planner_name or "navigation_mpc")
        planner_kwargs = dict(planner_config or {})
        if self.planner_name == "le_wm_cem" and world_model_path is not None:
            planner_kwargs.setdefault("checkpoint_path", world_model_path)
        self.planner = make_planner(self.planner_name, **planner_kwargs) if planner_name else NavigationMPCPlanner(**planner_kwargs)
        self._last_diagnostics: dict[str, Any] = {}
        self._stuck_steps = 0
        self._last_goal_distance: float | None = None

    def reset(self, scenario_info: Any) -> None:
        self.world_model.reset(scenario_info if isinstance(scenario_info, dict) else {"scenario": scenario_info})
        self.reference_agent.reset(scenario_info)
        self._last_diagnostics = {}
        self._stuck_steps = 0
        self._last_goal_distance = None

    def act(self, obs: Observation) -> Action:
        reference_action = self.reference_agent.act(obs)
        planning = self.planner.plan(obs, self.world_model, reference_action=reference_action)
        stabilized = _stabilize_action(planning.first_action, reference_action, obs)
        action, stuck_recovery = self._progress_filter(stabilized, reference_action, obs)
        self._last_diagnostics = {
            "agent": "world_model_direct",
            "planner": planning.metadata.get("planner") or self.planner_name,
            "world_model": self.world_model_name,
            "world_model_path": self.world_model_path,
            "best_cost": planning.best_cost,
            "planning": planning.metadata,
            "target_goal": [float(obs.goal[0]), float(obs.goal[1])],
            "route_used": False,
            "reference_action": _action_dict(reference_action),
            "executed_action": _action_dict(action),
            "stuck_recovery": stuck_recovery,
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        return None

    def _progress_filter(self, action: Action, reference_action: Action, obs: Observation) -> tuple[Action, bool]:
        state = obs.vehicle_state
        goal_distance = math.hypot(float(state.x) - float(obs.goal[0]), float(state.y) - float(obs.goal[1]))
        speed = max(0.0, float(state.speed))
        improving = self._last_goal_distance is None or goal_distance < self._last_goal_distance - 0.05
        if speed < 0.4 and not improving and action.throttle > 0.35:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        self._last_goal_distance = goal_distance
        if self._stuck_steps < 18:
            return action, False
        return (
            Action(
                steer=max(min(float(reference_action.steer), 0.18), -0.18),
                throttle=1.0,
                brake=0.0,
            ),
            True,
        )


def _stabilize_action(action: Action, reference_action: Action, obs: Observation) -> Action:
    speed = max(0.0, float(obs.vehicle_state.speed))
    steer = max(min(float(action.steer), 1.0), -1.0)
    throttle = max(min(float(action.throttle), 1.0), 0.0)
    brake = max(min(float(action.brake), 1.0), 0.0)
    turn_demand = max(abs(steer), abs(float(reference_action.steer)))
    if speed < 3.5:
        steer_conflict = abs(steer - float(reference_action.steer)) > 0.45
        planner_stalling = throttle < 0.35 or brake > 0.01 or abs(steer) > 0.75
        if steer_conflict or planner_stalling:
            steer_limit = 0.25 if speed < 1.0 else 0.5
            return Action(
                steer=max(min(float(reference_action.steer), steer_limit), -steer_limit),
                throttle=max(throttle, float(reference_action.throttle), 0.78),
                brake=0.0,
            )
    if speed > 6.0 and turn_demand > 0.35:
        throttle = min(throttle, 0.15)
        brake = max(brake, 0.12)
        steer = max(min(steer, 0.5), -0.5)
    return Action(steer=steer, throttle=throttle, brake=brake)
