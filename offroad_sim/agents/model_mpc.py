"""Model-scored MPC agent for bounded navigation tasks."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.algorithms import ScoreActionsRequest, make_algorithm
from offroad_sim.core import Action, Observation
from offroad_sim.planning.navigation_mpc import NavigationMPCPlanner
from offroad_sim.world_models import make_world_model


class ModelMPCAgent(OffroadAgent):
    """Control the vehicle through model-scored receding-horizon action search."""

    def __init__(
        self,
        route: list[tuple[float, float]] | None = None,
        waypoint_radius_m: float = 6.0,
        world_model_name: str = "simple_kinematic",
        world_model_path: str | Path | None = None,
        algorithm_name: str = "",
        algorithm_model_path: str | Path | None = None,
        planner_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.route = _normalize_route(route)
        self.waypoint_radius_m = float(waypoint_radius_m)
        self.cursor = 0
        self.world_model_name = str(world_model_name or "simple_kinematic")
        self.world_model_path = str(world_model_path) if world_model_path else None
        self.world_model = make_world_model(self.world_model_name, path=world_model_path) if world_model_path else make_world_model(self.world_model_name)
        self.algorithm_name = str(algorithm_name or "")
        self.algorithm_model_path = str(algorithm_model_path or world_model_path or "")
        self.algorithm = make_algorithm(self.algorithm_name) if self.algorithm_name else None
        if self.algorithm is not None and self.algorithm_model_path:
            self.algorithm.load(self.algorithm_model_path)
        self.reference_agent = RuleBasedGoalAgent(cruise_throttle=0.55)
        self.planner = NavigationMPCPlanner(**dict(planner_config or {}))
        self._last_diagnostics: dict[str, Any] = {}

    def reset(self, scenario_info: Any) -> None:
        if isinstance(scenario_info, dict):
            route = scenario_info.get("route") or scenario_info.get("beamng_route")
            if route:
                self.route = _normalize_route(route)
        self.cursor = 0
        self.world_model.reset(scenario_info if isinstance(scenario_info, dict) else {"scenario": scenario_info})
        self.reference_agent.reset(scenario_info)
        self._last_diagnostics = {}

    def act(self, obs: Observation) -> Action:
        if not self.route and obs.info.get("route"):
            self.route = _normalize_route(obs.info.get("route"))
            self.cursor = 0
        target = self._target_for(obs)
        routed_obs = replace(obs, goal=target) if target is not None else obs
        reference_action = self.reference_agent.act(routed_obs)
        planning = self.planner.plan(
            routed_obs,
            self.world_model,
            reference_action=reference_action,
            score_actions=(lambda candidates: self._score_candidates(routed_obs, candidates)) if self.algorithm is not None else None,
        )
        action = self._execution_filter(planning.first_action, reference_action, routed_obs)
        self._last_diagnostics = {
            "agent": "model_mpc",
            "planner": planning.metadata.get("planner"),
            "world_model": self.world_model_name,
            "world_model_path": self.world_model_path,
            "algorithm": self.algorithm_name or None,
            "algorithm_model_path": self.algorithm_model_path or None,
            "best_cost": planning.best_cost,
            "planning": planning.metadata,
            "route_length": len(self.route),
            "target_waypoint_index": self.cursor if self.route else None,
            "target_waypoint": list(target) if target is not None else None,
            "reference_action": _action_dict(reference_action),
            "executed_action": _action_dict(action),
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        return None

    def _score_candidates(self, observation: Observation, candidates: list[list[Action]]) -> list[float]:
        if self.algorithm is None:
            return [0.0 for _ in candidates]
        result = self.algorithm.score_actions(
            ScoreActionsRequest(observation=observation, action_candidates=candidates, task_info=observation.info)
        )
        return result.costs

    def _execution_filter(self, action: Action, reference_action: Action, obs: Observation) -> Action:
        speed = max(0.0, float(obs.vehicle_state.speed))
        steer = max(min(float(action.steer), 1.0), -1.0)
        throttle = max(min(float(action.throttle), 1.0), 0.0)
        brake = max(min(float(action.brake), 1.0), 0.0)
        if speed < 0.25 and throttle < 0.45:
            throttle = max(0.65, min(1.0, float(reference_action.throttle)))
            brake = 0.0
        if brake > 0.2 and throttle > 0.2:
            throttle = min(throttle, 0.2)
        return Action(steer=steer, throttle=throttle, brake=brake)

    def _target_for(self, obs: Observation) -> tuple[float, float] | None:
        if not self.route:
            return obs.goal
        state = obs.vehicle_state
        nearest_index = min(
            range(len(self.route)),
            key=lambda index: math.hypot(state.x - self.route[index][0], state.y - self.route[index][1]),
        )
        if nearest_index >= self.cursor:
            self.cursor = min(nearest_index + 1, len(self.route) - 1)
        if self.cursor < len(self.route) - 1:
            waypoint = self.route[self.cursor]
            if math.hypot(state.x - waypoint[0], state.y - waypoint[1]) <= self.waypoint_radius_m:
                self.cursor += 1
        return self.route[min(self.cursor, len(self.route) - 1)]


def _normalize_route(route: Any) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    if route is None:
        return rows
    for point in route:
        try:
            rows.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return rows


def _action_dict(action: Action) -> dict[str, float]:
    return {"steer": float(action.steer), "throttle": float(action.throttle), "brake": float(action.brake)}
