"""Route-free world-model MPC agent."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.agents.model_mpc import _action_dict, _goal_radius_from_info, _speed_steer_limit, _turn_speed_target
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
        allow_reverse_recovery: bool = False,
        reverse_recovery_after_steps: int = 96,
        local_subgoal_distance_m: float = 22.0,
        use_model_support_subgoals: bool = False,
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
        self.allow_reverse_recovery = bool(allow_reverse_recovery)
        self.reverse_recovery_after_steps = max(18, int(reverse_recovery_after_steps))
        self.local_subgoal_distance_m = max(1.0, float(local_subgoal_distance_m))
        self.use_model_support_subgoals = bool(use_model_support_subgoals)
        self._last_diagnostics: dict[str, Any] = {}
        self._stuck_steps = 0
        self._last_goal_distance: float | None = None
        self._last_progress_goal: tuple[float, float] | None = None
        self._last_position: tuple[float, float] | None = None
        self._goal_hold_latched = False
        self._experience_corridor_used = False
        self._model_support_subgoal_used = False

    def reset(self, scenario_info: Any) -> None:
        self.world_model.reset(scenario_info if isinstance(scenario_info, dict) else {"scenario": scenario_info})
        self.reference_agent.reset(scenario_info)
        self._last_diagnostics = {}
        self._stuck_steps = 0
        self._last_goal_distance = None
        self._last_progress_goal = None
        self._last_position = None
        self._goal_hold_latched = False
        self._experience_corridor_used = False
        self._model_support_subgoal_used = False

    def act(self, obs: Observation) -> Action:
        terminal_stop = self._terminal_stop_action(obs)
        if terminal_stop is not None:
            self._last_diagnostics = {
                "agent": "world_model_direct",
                "planner": self.planner_name,
                "world_model": self.world_model_name,
                "world_model_path": self.world_model_path,
                "target_goal": [float(obs.goal[0]), float(obs.goal[1])],
                "local_subgoal": [float(obs.goal[0]), float(obs.goal[1])],
                "planner_goal": [float(obs.goal[0]), float(obs.goal[1])],
                "route_used": False,
                "experience_corridor_used": False,
                "model_support_subgoal_used": False,
                "goal_stop": True,
                "goal_hold_latched": self._goal_hold_latched,
                "executed_action": _action_dict(terminal_stop),
            }
            return terminal_stop
        local_subgoal = self._local_subgoal(obs)
        planning_obs = _observation_with_goal(obs, local_subgoal)
        reference_action = self.reference_agent.act(planning_obs)
        planning = self.planner.plan(planning_obs, self.world_model, reference_action=reference_action)
        stabilized = _stabilize_action(planning.first_action, reference_action, obs)
        action, stuck_recovery = self._progress_filter(stabilized, reference_action, planning_obs)
        self._last_diagnostics = {
            "agent": "world_model_direct",
            "planner": planning.metadata.get("planner") or self.planner_name,
            "world_model": self.world_model_name,
            "world_model_path": self.world_model_path,
            "best_cost": planning.best_cost,
            "planning": planning.metadata,
            "target_goal": [float(obs.goal[0]), float(obs.goal[1])],
            "local_subgoal": [float(local_subgoal[0]), float(local_subgoal[1])],
            "planner_goal": [float(planning_obs.goal[0]), float(planning_obs.goal[1])],
            "route_used": False,
            "experience_corridor_used": self._experience_corridor_used,
            "model_support_subgoal_used": self._model_support_subgoal_used,
            "reference_action": _action_dict(reference_action),
            "executed_action": _action_dict(action),
            "stuck_recovery": stuck_recovery,
            "allow_reverse_recovery": self.allow_reverse_recovery,
            "reverse_recovery_after_steps": self.reverse_recovery_after_steps,
            "goal_stop": False,
            "goal_hold_latched": self._goal_hold_latched,
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        return None

    def _progress_filter(self, action: Action, reference_action: Action, obs: Observation) -> tuple[Action, bool]:
        state = obs.vehicle_state
        position = (float(state.x), float(state.y))
        progress_goal = (float(obs.goal[0]), float(obs.goal[1]))
        if self._last_progress_goal is None or math.hypot(progress_goal[0] - self._last_progress_goal[0], progress_goal[1] - self._last_progress_goal[1]) > 0.75:
            self._last_goal_distance = None
            self._last_progress_goal = progress_goal
        goal_distance = math.hypot(float(state.x) - progress_goal[0], float(state.y) - progress_goal[1])
        speed = max(0.0, float(state.speed))
        improving = self._last_goal_distance is None or goal_distance < self._last_goal_distance - 0.05
        moved = self._last_position is None or math.hypot(position[0] - self._last_position[0], position[1] - self._last_position[1]) >= 0.03
        self._last_position = position
        physically_stalled = speed < 0.08 and not moved
        if speed < 0.4 and action.throttle > 0.35 and (not improving or physically_stalled):
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        self._last_goal_distance = goal_distance
        if self._stuck_steps < 18:
            return action, False
        reference_steer = max(min(float(reference_action.steer), 1.0), -1.0)
        turn_demand = abs(reference_steer)
        recovery_phase = ((self._stuck_steps - 18) // 12) % 3
        reverse_allowed = self.allow_reverse_recovery and self._stuck_steps >= self.reverse_recovery_after_steps
        if turn_demand > 0.35 and recovery_phase == 0:
            if not reverse_allowed:
                return (
                    Action(
                        steer=max(min(reference_steer, 0.55), -0.55),
                        throttle=min(max(float(reference_action.throttle), 0.28), 0.55),
                        brake=0.0,
                        gear=None,
                    ),
                    True,
                )
            return (
                Action(steer=0.0, throttle=0.55, brake=0.0, gear=-1),
                True,
            )
        if turn_demand > 0.35 and recovery_phase == 2:
            return (
                Action(
                    steer=max(min(reference_steer, 0.45), -0.45),
                    throttle=0.75,
                    brake=0.0,
                    gear=None,
                ),
                True,
            )
        if turn_demand > 0.75:
            return (
                Action(
                    steer=max(min(reference_steer, 0.9), -0.9),
                    throttle=min(max(float(reference_action.throttle), 0.55), 0.7),
                    brake=0.0,
                    gear=None,
                ),
                True,
            )
        if turn_demand > 0.35:
            return (
                Action(
                    steer=max(min(reference_steer, 0.65), -0.65),
                    throttle=min(max(float(reference_action.throttle), 0.65), 0.8),
                    brake=0.0,
                    gear=None,
                ),
                True,
            )
        if recovery_phase == 0:
            recovery_step = (self._stuck_steps - 18) % 12
            if not reverse_allowed:
                if recovery_step < 3:
                    return (
                        Action(
                            steer=max(min(reference_steer, 0.18), -0.18),
                            throttle=0.0,
                            brake=0.28,
                            gear=None,
                        ),
                        True,
                    )
                return (
                    Action(
                        steer=max(min(reference_steer, 0.25), -0.25),
                        throttle=0.38,
                        brake=0.0,
                        gear=None,
                    ),
                    True,
                )
            return (Action(steer=0.0, throttle=0.55, brake=0.0, gear=-1), True)
        return (
            Action(
                steer=max(min(reference_steer, 0.25), -0.25),
                throttle=1.0,
                brake=0.0,
                gear=None,
            ),
            True,
        )

    def _terminal_stop_action(self, obs: Observation) -> Action | None:
        radius = _goal_radius_from_info(obs.info)
        if radius <= 0.0:
            return None
        state = obs.vehicle_state
        distance = math.hypot(float(state.x) - float(obs.goal[0]), float(state.y) - float(obs.goal[1]))
        if distance <= radius:
            self._goal_hold_latched = True
        if not self._goal_hold_latched:
            return None
        self._stuck_steps = 0
        self._last_goal_distance = distance
        self._last_progress_goal = (float(obs.goal[0]), float(obs.goal[1]))
        self._last_position = (float(state.x), float(state.y))
        return Action(steer=0.0, throttle=0.0, brake=1.0, gear=0)

    def _local_subgoal(self, obs: Observation) -> tuple[float, float]:
        self._experience_corridor_used = False
        self._model_support_subgoal_used = False
        state = obs.vehicle_state
        start = (float(state.x), float(state.y))
        goal = (float(obs.goal[0]), float(obs.goal[1]))
        experience_subgoal = _experience_route_subgoal(
            obs.info,
            start=start,
            lookahead_m=self.local_subgoal_distance_m,
            goal=goal,
        )
        if experience_subgoal is not None:
            self._experience_corridor_used = True
            return experience_subgoal
        if self.use_model_support_subgoals:
            support_subgoal = _support_route_subgoal(
                _model_support_route(self.world_model),
                info=obs.info,
                start=start,
                lookahead_m=self.local_subgoal_distance_m,
                goal=goal,
            )
            if support_subgoal is not None:
                self._model_support_subgoal_used = True
                return support_subgoal
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= self.local_subgoal_distance_m:
            return goal
        ux = dx / distance
        uy = dy / distance
        polygon = _navigation_polygon(obs.info)
        turn_arc_subgoal = self._turn_arc_subgoal(obs, start, goal, polygon)
        if turn_arc_subgoal is not None:
            return turn_arc_subgoal
        recovery_subgoal = self._recovery_subgoal(start, goal, polygon)
        if recovery_subgoal is not None:
            return recovery_subgoal
        for scale in (1.0, 0.8, 0.6, 0.4, 0.25):
            step = self.local_subgoal_distance_m * scale
            candidate = (start[0] + ux * step, start[1] + uy * step)
            if not polygon or _point_in_polygon(candidate, polygon):
                return candidate
        return start

    def _turn_arc_subgoal(
        self,
        obs: Observation,
        start: tuple[float, float],
        goal: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= self.local_subgoal_distance_m:
            return None
        target_heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(target_heading - float(obs.vehicle_state.yaw))
        if abs(heading_error) < 1.15:
            return None
        turn = math.copysign(min(abs(heading_error), 0.75), heading_error)
        for step_scale in (0.9, 0.7, 0.5):
            heading = float(obs.vehicle_state.yaw) + turn
            step = self.local_subgoal_distance_m * step_scale
            candidate = (start[0] + math.cos(heading) * step, start[1] + math.sin(heading) * step)
            if not polygon or _point_in_polygon(candidate, polygon):
                return candidate
        return None

    def _recovery_subgoal(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        if self._stuck_steps < 18:
            return None
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return None
        ux = dx / distance
        uy = dy / distance
        side_x = -uy
        side_y = ux
        phase = ((self._stuck_steps - 18) // 12) % 4
        side_order = (1.0, -1.0) if phase in {0, 2} else (-1.0, 1.0)
        if self._stuck_steps >= 30:
            forward_scales = (0.2, 0.35, 0.55, 0.0)
            lateral_scales = (1.15, 0.9, 0.65, 1.35)
        else:
            forward_scales = (0.9, 0.65, 0.45)
            lateral_scales = (0.45, 0.7, 0.3)
        for side in side_order:
            for forward_scale in forward_scales:
                for lateral_scale in lateral_scales:
                    forward = min(distance, self.local_subgoal_distance_m * forward_scale)
                    lateral = self.local_subgoal_distance_m * lateral_scale * side
                    candidate = (
                        start[0] + ux * forward + side_x * lateral,
                        start[1] + uy * forward + side_y * lateral,
                    )
                    if not polygon or _point_in_polygon(candidate, polygon):
                        return candidate
        return None


def _observation_with_goal(obs: Observation, goal: tuple[float, float]) -> Observation:
    return Observation(
        timestamp=obs.timestamp,
        vehicle_state=obs.vehicle_state,
        goal=(float(goal[0]), float(goal[1])),
        front_rgb=obs.front_rgb,
        depth=obs.depth,
        lidar_points=obs.lidar_points,
        local_bev=obs.local_bev,
        terrain_map=obs.terrain_map,
        info=obs.info,
    )


def _navigation_polygon(info: dict[str, Any]) -> list[tuple[float, float]]:
    task = info.get("navigation_region", {}) if isinstance(info, dict) else {}
    region = task.get("region", {}) if isinstance(task, dict) else {}
    raw = region.get("polygon", []) if isinstance(region, dict) else []
    polygon: list[tuple[float, float]] = []
    for point in raw or []:
        try:
            polygon.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return polygon if len(polygon) >= 3 else []


def _experience_route_subgoal(
    info: dict[str, Any],
    *,
    start: tuple[float, float],
    lookahead_m: float,
    goal: tuple[float, float],
) -> tuple[float, float] | None:
    route = _experience_route(info)
    return _support_route_subgoal(route, info=info, start=start, lookahead_m=lookahead_m, goal=goal)


def _support_route_subgoal(
    route: list[tuple[float, float]],
    *,
    info: dict[str, Any],
    start: tuple[float, float],
    lookahead_m: float,
    goal: tuple[float, float],
) -> tuple[float, float] | None:
    if len(route) < 2:
        return None
    polygon = _navigation_polygon(info)
    nearest_index = min(range(len(route)), key=lambda index: math.hypot(route[index][0] - start[0], route[index][1] - start[1]))
    travelled = 0.0
    previous = route[nearest_index]
    for point in route[nearest_index + 1 :]:
        segment = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if travelled + segment + 1e-6 >= lookahead_m:
            remaining = max(0.0, float(lookahead_m) - travelled)
            ratio = 1.0 if segment <= 1e-9 else min(1.0, remaining / segment)
            candidate = (
                previous[0] + (point[0] - previous[0]) * ratio,
                previous[1] + (point[1] - previous[1]) * ratio,
            )
            if not polygon or _point_in_polygon(candidate, polygon):
                return candidate
            if not polygon or _point_in_polygon(point, polygon):
                return point
        travelled += segment
        previous = point
    final = route[-1]
    if math.hypot(final[0] - goal[0], final[1] - goal[1]) <= max(lookahead_m, 1.0):
        return goal
    return final if not polygon or _point_in_polygon(final, polygon) else None


def _model_support_route(world_model: Any) -> list[tuple[float, float]]:
    metadata = getattr(world_model, "metadata", {})
    raw = metadata.get("support_points", []) if isinstance(metadata, dict) else []
    route: list[tuple[float, float]] = []
    for point in raw or []:
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            route.append((x, y))
    return route


def _experience_route(info: dict[str, Any]) -> list[tuple[float, float]]:
    raw: Any = []
    if isinstance(info, dict):
        if isinstance(info.get("experience_route"), list):
            raw = info.get("experience_route")
        else:
            task = info.get("navigation_region", {})
            if isinstance(task, dict) and isinstance(task.get("experience_route"), list):
                raw = task.get("experience_route")
    route: list[tuple[float, float]] = []
    for point in raw or []:
        try:
            route.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return route


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        if ((current[1] > y) != (previous[1] > y)) and (
            x < (previous[0] - current[0]) * (y - current[1]) / ((previous[1] - current[1]) or 1e-12) + current[0]
        ):
            inside = not inside
        j = i
    return inside


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _stabilize_action(action: Action, reference_action: Action, obs: Observation) -> Action:
    speed = max(0.0, float(obs.vehicle_state.speed))
    steer = max(min(float(action.steer), 1.0), -1.0)
    throttle = max(min(float(action.throttle), 1.0), 0.0)
    brake = max(min(float(action.brake), 1.0), 0.0)
    gear = int(action.gear) if action.gear is not None else None
    reference_steer = max(min(float(reference_action.steer), 1.0), -1.0)
    turn_demand = max(abs(steer), abs(reference_steer))
    sharp_turn = abs(reference_steer) > 0.75
    if speed < 3.5:
        steer_conflict = abs(steer - reference_steer) > 0.45
        planner_stalling = throttle < 0.35 or brake > 0.01 or abs(steer) > 0.75
        if steer_conflict or planner_stalling:
            if sharp_turn:
                steer_limit = 0.35 if speed < 1.0 else 0.5
                throttle_floor = 0.78 if speed < 1.0 else 0.58
                throttle_cap = 0.9 if speed < 1.0 else 0.75
                return Action(
                    steer=max(min(reference_steer, steer_limit), -steer_limit),
                    throttle=min(max(throttle, float(reference_action.throttle), throttle_floor), throttle_cap),
                    brake=0.0,
                    gear=gear,
                )
            steer_limit = 0.25 if speed < 1.0 else 0.5
            return Action(
                steer=max(min(reference_steer, steer_limit), -steer_limit),
                throttle=max(throttle, float(reference_action.throttle), 0.78),
                brake=0.0,
                gear=gear,
            )
    if abs(reference_steer) > 0.35 and steer * reference_steer < -0.05:
        steer_limit = _speed_steer_limit(speed, sharp_turn=sharp_turn)
        steer = max(min(reference_steer, steer_limit), -steer_limit)
        turn_demand = max(abs(steer), abs(reference_steer))
    if turn_demand > 0.35:
        steer_limit = _speed_steer_limit(speed, sharp_turn=sharp_turn)
        steer = max(min(steer, steer_limit), -steer_limit)
        target_speed = _turn_speed_target(turn_demand)
        if speed > target_speed:
            overspeed = speed - target_speed
            throttle = 0.0 if overspeed >= 1.0 else min(throttle, 0.15)
            brake = max(brake, min(0.45, 0.08 + 0.08 * overspeed))
        elif speed > target_speed - 0.4:
            throttle = min(throttle, 0.2)
    if brake > 0.2 and throttle > 0.2:
        throttle = min(throttle, 0.2)
    return Action(steer=steer, throttle=throttle, brake=brake, gear=gear)
