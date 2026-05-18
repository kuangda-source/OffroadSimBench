"""Route-following world-model agent for visible simulator demos."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.agents.world_model import WorldModelAgent
from offroad_sim.core import Action, Observation


class RouteWorldModelAgent(OffroadAgent):
    """Follow route waypoints while keeping model/planner selection switchable."""

    def __init__(
        self,
        route: list[tuple[float, float]] | None = None,
        waypoint_radius_m: float = 6.0,
        **world_model_kwargs: Any,
    ) -> None:
        self.route = _normalize_route(route)
        self.waypoint_radius_m = float(waypoint_radius_m)
        self.cursor = 0
        self.execution_mode = str(world_model_kwargs.pop("execution_mode", "model_guided_route_tracker"))
        world_model_kwargs.pop("seed", None)
        self.inner = WorldModelAgent(**world_model_kwargs)
        self.progress_agent = RuleBasedGoalAgent(cruise_throttle=0.55)
        self._last_diagnostics: dict[str, Any] = {}
        self._stuck_steps = 0

    def reset(self, scenario_info: Any) -> None:
        if isinstance(scenario_info, dict):
            route = scenario_info.get("route") or scenario_info.get("beamng_route")
            if route:
                self.route = _normalize_route(route)
        self.cursor = 0
        self._stuck_steps = 0
        self.inner.reset(scenario_info)
        self._last_diagnostics = {}

    def act(self, obs: Observation) -> Action:
        if not self.route and obs.info.get("route"):
            self.route = _normalize_route(obs.info.get("route"))
            self.cursor = 0
        target = self._target_for(obs)
        routed_obs = replace(obs, goal=target) if target is not None else obs
        planner_action = self.inner.act(routed_obs)
        reference_action = self.progress_agent.act(routed_obs)
        if self.execution_mode in {"planner_direct", "direct"}:
            action, progress_guard = self._apply_progress_guard(planner_action, reference_action, routed_obs)
            stuck_recovery = False
            controller_name = "planner_direct"
        else:
            action, progress_guard, stuck_recovery = self._route_tracker_action(planner_action, reference_action, routed_obs)
            controller_name = "model_guided_route_tracker"
        inner_diagnostics = self.inner.diagnostics()
        self._last_diagnostics = {
            **inner_diagnostics,
            "route_length": len(self.route),
            "target_waypoint_index": self.cursor if self.route else None,
            "target_waypoint": list(target) if target is not None else None,
            "progress_guard": progress_guard,
            "stuck_recovery": stuck_recovery,
            "execution_controller": controller_name,
            "planner_action": _action_dict(planner_action),
            "reference_action": _action_dict(reference_action),
            "executed_action": _action_dict(action),
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        self.inner.close()

    def _apply_progress_guard(self, action: Action, reference_action: Action, obs: Observation) -> tuple[Action, bool]:
        speed = max(0.0, float(obs.vehicle_state.speed))
        if speed > 4.0:
            return action, False
        steer_conflict = abs(float(action.steer) - float(reference_action.steer)) > 0.65
        if speed < 4.0 and (action.brake > 0.01 or action.throttle < 0.55 or steer_conflict):
            steer_limit = _low_speed_steer_limit(speed)
            return (
                Action(
                    steer=max(min(reference_action.steer, steer_limit), -steer_limit),
                    throttle=max(action.throttle, reference_action.throttle, 0.55),
                    brake=0.0,
                ),
                True,
            )
        if action.throttle >= 0.15 and action.brake <= 0.45:
            return action, False
        return (
            Action(
                steer=reference_action.steer,
                throttle=max(action.throttle, min(reference_action.throttle, 0.45)),
                brake=min(max(action.brake, 0.0), 0.15),
            ),
            True,
        )

    def _route_tracker_action(
        self,
        planner_action: Action,
        reference_action: Action,
        obs: Observation,
    ) -> tuple[Action, bool, bool]:
        state = obs.vehicle_state
        speed = max(0.0, float(state.speed))
        goal_x, goal_y = obs.goal
        target_heading = math.atan2(goal_y - state.y, goal_x - state.x)
        heading_error = _wrap_angle(target_heading - state.yaw)
        steer_limit = _low_speed_steer_limit(speed) if speed < 4.0 else 0.85
        steer = max(min(heading_error / 0.75, steer_limit), -steer_limit)

        if abs(heading_error) < 0.35:
            target_speed = 7.0
        elif abs(heading_error) < 0.8:
            target_speed = 5.0
        else:
            target_speed = 3.0

        throttle = 0.72 if speed < target_speed else 0.22
        brake = 0.0 if speed <= target_speed + 1.5 else 0.12
        if speed < 0.6:
            throttle = max(throttle, 0.85)

        if speed < 0.35 and throttle > 0.4:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        stuck_recovery = self._stuck_steps >= 12
        if stuck_recovery:
            throttle = 1.0
            brake = 0.0
            steer = max(min(steer, 0.2), -0.2)

        action = Action(steer=steer, throttle=throttle, brake=brake)
        progress_guard = (
            stuck_recovery
            or abs(action.steer - planner_action.steer) > 0.2
            or abs(action.throttle - planner_action.throttle) > 0.2
            or abs(action.brake - planner_action.brake) > 0.05
        )
        return action, progress_guard, stuck_recovery

    def _target_for(self, obs: Observation) -> tuple[float, float] | None:
        if not self.route:
            return None
        state = obs.vehicle_state
        nearest_index = min(
            range(len(self.route)),
            key=lambda index: math.hypot(state.x - self.route[index][0], state.y - self.route[index][1]),
        )
        advanced_by_nearest = False
        if nearest_index >= self.cursor:
            next_cursor = min(nearest_index + 1, len(self.route) - 1)
            advanced_by_nearest = next_cursor != self.cursor
            self.cursor = next_cursor
        if not advanced_by_nearest and self.cursor < len(self.route) - 1:
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


def _low_speed_steer_limit(speed: float) -> float:
    if speed < 1.0:
        return 0.35
    if speed < 2.5:
        return 0.5
    return 0.65


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _action_dict(action: Action) -> dict[str, float]:
    return {"steer": float(action.steer), "throttle": float(action.throttle), "brake": float(action.brake)}
