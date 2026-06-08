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
        route_lookahead_m: float = 8.0,
        world_model_name: str = "simple_kinematic",
        world_model_path: str | Path | None = None,
        algorithm_name: str = "",
        algorithm_model_path: str | Path | None = None,
        planner_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.route = _normalize_route(route)
        self.waypoint_radius_m = float(waypoint_radius_m)
        self.route_lookahead_m = max(float(route_lookahead_m), self.waypoint_radius_m)
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
        self._stuck_steps = 0
        self._stuck_recovery = False

    def reset(self, scenario_info: Any) -> None:
        if isinstance(scenario_info, dict):
            route = scenario_info.get("route") or scenario_info.get("beamng_route")
            if route:
                self.route = _normalize_route(route)
        self.cursor = 0
        self.world_model.reset(scenario_info if isinstance(scenario_info, dict) else {"scenario": scenario_info})
        self.reference_agent.reset(scenario_info)
        self._last_diagnostics = {}
        self._stuck_steps = 0
        self._stuck_recovery = False

    def act(self, obs: Observation) -> Action:
        terminal_action = self._terminal_stop_action(obs)
        if terminal_action is not None:
            self._last_diagnostics = {
                "agent": "model_mpc",
                "world_model": self.world_model_name,
                "world_model_path": self.world_model_path,
                "algorithm": self.algorithm_name or None,
                "algorithm_model_path": self.algorithm_model_path or None,
                "route_length": len(self.route),
                "terminal_stop": True,
                "target_goal": [float(obs.goal[0]), float(obs.goal[1])],
                "executed_action": _action_dict(terminal_action),
            }
            return terminal_action
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
            "stuck_steps": self._stuck_steps,
            "stuck_recovery": self._stuck_recovery,
            "terminal_stop": False,
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        if not self._last_diagnostics:
            return {"stuck_steps": self._stuck_steps, "stuck_recovery": self._stuck_recovery}
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

    def _terminal_stop_action(self, obs: Observation) -> Action | None:
        radius = _goal_radius_from_info(obs.info)
        if radius <= 0.0:
            return None
        state = obs.vehicle_state
        distance = math.hypot(float(state.x) - float(obs.goal[0]), float(state.y) - float(obs.goal[1]))
        if distance > radius:
            return None
        self._stuck_steps = 0
        self._stuck_recovery = False
        return Action(steer=0.0, throttle=0.0, brake=1.0)

    def _execution_filter(self, action: Action, reference_action: Action, obs: Observation) -> Action:
        speed = max(0.0, float(obs.vehicle_state.speed))
        steer = max(min(float(action.steer), 1.0), -1.0)
        throttle = max(min(float(action.throttle), 1.0), 0.0)
        brake = max(min(float(action.brake), 1.0), 0.0)
        reference_steer = float(reference_action.steer)
        sharp_turn = abs(reference_steer) > 0.75
        low_speed_turn = speed < 0.35 and abs(reference_steer) > 0.5
        low_speed_no_progress = speed < 0.25 and max(throttle, float(reference_action.throttle)) > 0.4
        if low_speed_turn or low_speed_no_progress:
            self._stuck_steps += 1
        elif speed > 0.8:
            self._stuck_steps = 0
        self._stuck_recovery = self._stuck_steps >= 12
        if speed < 0.25 and throttle <= 0.45:
            throttle = max(0.65, min(1.0, float(reference_action.throttle)))
            brake = 0.0
        if self._stuck_recovery:
            throttle = max(throttle, 1.0)
            brake = 0.0
            if speed < 0.2:
                steer = _recovery_steer(reference_steer, self._stuck_steps)
            else:
                recovery_steer_limit = 1.0 if sharp_turn else 0.55
                steer = max(min(reference_steer, recovery_steer_limit), -recovery_steer_limit)
        else:
            steer_limit = _speed_steer_limit(speed, sharp_turn=sharp_turn)
            if abs(steer) > steer_limit:
                steer = max(min(steer, steer_limit), -steer_limit)
            turn_demand = max(abs(reference_steer), abs(steer))
            target_speed = _turn_speed_target(turn_demand)
            if turn_demand >= 0.35 and speed > target_speed:
                overspeed = speed - target_speed
                throttle = 0.0 if overspeed >= 1.0 else min(throttle, 0.15)
                brake = max(brake, min(0.45, 0.08 + 0.08 * overspeed))
            elif turn_demand >= 0.35 and speed > target_speed - 0.4:
                throttle = min(throttle, 0.2)
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
        lookahead_index = self._lookahead_index(nearest_index)
        if lookahead_index > self.cursor:
            self.cursor = lookahead_index
        if self.cursor < len(self.route) - 1:
            waypoint = self.route[self.cursor]
            if math.hypot(state.x - waypoint[0], state.y - waypoint[1]) <= self.waypoint_radius_m:
                self.cursor += 1
        return self.route[min(self.cursor, len(self.route) - 1)]

    def _lookahead_index(self, nearest_index: int) -> int:
        target_index = min(nearest_index + 1, len(self.route) - 1)
        distance = 0.0
        for index in range(nearest_index, len(self.route) - 1):
            a = self.route[index]
            b = self.route[index + 1]
            distance += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            target_index = index + 1
            if distance >= self.route_lookahead_m:
                break
        return target_index


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


def _speed_steer_limit(speed: float, *, sharp_turn: bool = False) -> float:
    if speed >= 7.0:
        return 0.42
    if speed >= 5.0:
        return 0.55 if sharp_turn else 0.5
    if speed >= 3.0:
        return 0.75 if sharp_turn else 0.65
    return 1.0


def _turn_speed_target(turn_demand: float) -> float:
    turn = abs(float(turn_demand))
    if turn >= 0.75:
        return 3.2
    if turn >= 0.55:
        return 4.4
    if turn >= 0.35:
        return 5.6
    return 8.0


def _recovery_steer(reference_steer: float, stuck_steps: int) -> float:
    direction = 1.0 if reference_steer >= 0.0 else -1.0
    phase = ((max(0, int(stuck_steps)) - 12) // 20) % 3
    if phase == 0:
        return 0.9 * direction
    if phase == 1:
        return 1.0 * direction
    return 0.55 * direction


def _goal_radius_from_info(info: dict[str, Any]) -> float:
    candidates: list[Any] = [
        info.get("goal_radius"),
        info.get("success_radius_m"),
    ]
    navigation_region = info.get("navigation_region")
    if isinstance(navigation_region, dict):
        candidates.append(navigation_region.get("goal_radius"))
        goal = navigation_region.get("goal")
        if isinstance(goal, dict):
            candidates.append(goal.get("radius"))
    for value in candidates:
        try:
            radius = float(value)
        except (TypeError, ValueError):
            continue
        if radius > 0.0:
            return radius
    return 0.0


def _action_dict(action: Action) -> dict[str, float]:
    return {"steer": float(action.steer), "throttle": float(action.throttle), "brake": float(action.brake)}
