"""Route-free world-model MPC agent."""

from __future__ import annotations

import heapq
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


SUPPORT_ROUTE_BRIDGE_MAX_M = 32.0
SUPPORT_ROUTE_MIN_GOAL_PROGRESS_M = 1.0
SUPPORT_FIELD_NEAR_GAIN = 2.5
SUPPORT_FIELD_MIN_STEP_FRACTION = 0.33
SUPPORT_FIELD_USE_MARGIN_M = 0.5
STALL_MOVEMENT_EPS_M = 0.12


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
        use_model_support_field_subgoals: bool = False,
        use_model_support_graph_subgoals: bool = False,
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
        self.use_model_support_field_subgoals = bool(use_model_support_field_subgoals)
        self.use_model_support_graph_subgoals = bool(use_model_support_graph_subgoals)
        self._last_diagnostics: dict[str, Any] = {}
        self._stuck_steps = 0
        self._last_goal_distance: float | None = None
        self._last_progress_goal: tuple[float, float] | None = None
        self._last_position: tuple[float, float] | None = None
        self._goal_hold_latched = False
        self._experience_corridor_used = False
        self._model_support_subgoal_used = False
        self._model_support_field_subgoal_used = False
        self._model_support_graph_subgoal_used = False

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
        self._model_support_field_subgoal_used = False
        self._model_support_graph_subgoal_used = False

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
                "model_support_field_subgoal_used": False,
                "model_support_graph_subgoal_used": False,
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
            "model_support_field_subgoal_used": self._model_support_field_subgoal_used,
            "model_support_graph_subgoal_used": self._model_support_graph_subgoal_used,
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
        moved = self._last_position is None or math.hypot(position[0] - self._last_position[0], position[1] - self._last_position[1]) >= STALL_MOVEMENT_EPS_M
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
        self._model_support_field_subgoal_used = False
        self._model_support_graph_subgoal_used = False
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
        if self.use_model_support_graph_subgoals:
            support_graph_subgoal = _support_graph_subgoal(
                _model_support_graph(self.world_model),
                info=obs.info,
                start=start,
                lookahead_m=self.local_subgoal_distance_m,
                goal=goal,
            )
            if support_graph_subgoal is not None:
                self._model_support_graph_subgoal_used = True
                return support_graph_subgoal
        if self.use_model_support_subgoals:
            support_subgoal = _support_routes_subgoal(
                _model_support_routes(self.world_model),
                info=obs.info,
                start=start,
                lookahead_m=self.local_subgoal_distance_m,
                goal=goal,
            )
            if support_subgoal is not None:
                self._model_support_subgoal_used = True
                return support_subgoal
        if self.use_model_support_field_subgoals:
            support_field_subgoal = _support_field_subgoal(
                _model_support_points(self.world_model),
                info=obs.info,
                start=start,
                lookahead_m=self.local_subgoal_distance_m,
                goal=goal,
                stuck_steps=self._stuck_steps,
            )
            if support_field_subgoal is not None:
                self._model_support_field_subgoal_used = True
                return support_field_subgoal
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= self.local_subgoal_distance_m:
            return goal
        ux = dx / distance
        uy = dy / distance
        polygon = _navigation_polygon(obs.info)
        recovery_subgoal = self._recovery_subgoal(start, goal, polygon)
        if recovery_subgoal is not None:
            return recovery_subgoal
        for scale in (1.0, 0.8, 0.6, 0.4, 0.25):
            step = self.local_subgoal_distance_m * scale
            candidate = (start[0] + ux * step, start[1] + uy * step)
            if not polygon or _point_in_polygon(candidate, polygon):
                return candidate
        turn_arc_subgoal = self._turn_arc_subgoal(obs, start, goal, polygon)
        if turn_arc_subgoal is not None:
            return turn_arc_subgoal
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


def _support_routes_subgoal(
    routes: list[list[tuple[float, float]]],
    *,
    info: dict[str, Any],
    start: tuple[float, float],
    lookahead_m: float,
    goal: tuple[float, float],
) -> tuple[float, float] | None:
    if not routes:
        return None
    stitched = _stitched_support_route(
        routes,
        start=start,
        goal=goal,
        max_bridge_m=max(SUPPORT_ROUTE_BRIDGE_MAX_M, float(lookahead_m) * 2.0),
    )
    if stitched:
        subgoal = _support_route_subgoal(stitched, info=info, start=start, lookahead_m=lookahead_m, goal=goal)
        if subgoal is not None:
            return subgoal
    candidates = sorted(
        routes,
        key=lambda route: min(math.hypot(point[0] - start[0], point[1] - start[1]) for point in route),
    )
    for route in candidates:
        subgoal = _support_route_subgoal(route, info=info, start=start, lookahead_m=lookahead_m, goal=goal)
        if subgoal is not None:
            return subgoal
    return None


def _support_field_subgoal(
    points: list[tuple[float, float]],
    *,
    info: dict[str, Any],
    start: tuple[float, float],
    lookahead_m: float,
    goal: tuple[float, float],
    stuck_steps: int,
) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    polygon = _navigation_polygon(info)
    goal_dx = goal[0] - start[0]
    goal_dy = goal[1] - start[1]
    goal_distance = math.hypot(goal_dx, goal_dy)
    if goal_distance <= max(1e-6, float(lookahead_m) * 0.5):
        return None

    direct_candidate = _candidate_toward(start, goal, min(float(lookahead_m), goal_distance))
    direct_support_distance = _nearest_point_distance(direct_candidate, points)
    min_candidate_distance = max(2.0, float(lookahead_m) * SUPPORT_FIELD_MIN_STEP_FRACTION)
    min_goal_progress = max(0.5, float(lookahead_m) * 0.05)
    candidates: list[tuple[float, float]] = []
    for point in points:
        point_distance = math.hypot(point[0] - start[0], point[1] - start[1])
        if point_distance < min_candidate_distance:
            continue
        if point_distance <= float(lookahead_m) * 1.15:
            candidate = point
        else:
            candidate = _candidate_toward(start, point, float(lookahead_m))
        if polygon and not _point_in_polygon(candidate, polygon):
            continue
        candidate_distance = math.hypot(candidate[0] - start[0], candidate[1] - start[1])
        if candidate_distance < min_candidate_distance:
            continue
        alignment = ((candidate[0] - start[0]) * goal_dx + (candidate[1] - start[1]) * goal_dy) / (candidate_distance * goal_distance)
        if alignment < -0.15:
            continue
        if math.hypot(candidate[0] - goal[0], candidate[1] - goal[1]) > goal_distance - min_goal_progress:
            continue
        candidates.append(candidate)
    if not candidates:
        return None

    def score(candidate: tuple[float, float]) -> tuple[float, float, float]:
        support_distance = _nearest_point_distance(candidate, points)
        goal_after_candidate = math.hypot(candidate[0] - goal[0], candidate[1] - goal[1])
        short_step_penalty = max(0.0, float(lookahead_m) * 0.35 - math.hypot(candidate[0] - start[0], candidate[1] - start[1]))
        stuck_bias = 0.6 if int(stuck_steps) >= 18 else 1.0
        total = goal_after_candidate + SUPPORT_FIELD_NEAR_GAIN * stuck_bias * support_distance + 0.35 * short_step_penalty
        return (total, support_distance, goal_after_candidate)

    best = min(candidates, key=score)
    if _nearest_point_distance(best, points) + SUPPORT_FIELD_USE_MARGIN_M < direct_support_distance or int(stuck_steps) >= 18:
        return best
    return None


def _support_graph_subgoal(
    graph: dict[str, Any],
    *,
    info: dict[str, Any],
    start: tuple[float, float],
    lookahead_m: float,
    goal: tuple[float, float],
) -> tuple[float, float] | None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if len(nodes) < 2 or not edges:
        return None
    polygon = _navigation_polygon(info)
    start_candidates = _nearest_node_indices(
        nodes,
        start,
        limit=8,
        max_distance=max(float(lookahead_m) * 2.5, SUPPORT_ROUTE_BRIDGE_MAX_M),
    )
    if not start_candidates:
        return None
    goal_candidates = _nearest_node_indices(nodes, goal, limit=8)
    if not goal_candidates:
        return None

    best_path: list[int] = []
    best_cost = float("inf")
    goal_set = set(goal_candidates)
    for start_index in start_candidates:
        path, cost = _support_graph_shortest_path(nodes, edges, start_index=start_index, goal_indices=goal_set)
        if len(path) < 2:
            continue
        start_cost = math.hypot(nodes[start_index][0] - start[0], nodes[start_index][1] - start[1])
        goal_cost = min(math.hypot(nodes[path[-1]][0] - goal[0], nodes[path[-1]][1] - goal[1]), float(lookahead_m))
        total_cost = float(cost) + start_cost + goal_cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_path = path
    if not best_path:
        return None

    path_points = [nodes[index] for index in best_path]
    return _support_path_local_subgoal(path_points, start=start, lookahead_m=lookahead_m, goal=goal, polygon=polygon)


def _nearest_node_indices(
    nodes: list[tuple[float, float]],
    target: tuple[float, float],
    *,
    limit: int,
    max_distance: float = float("inf"),
) -> list[int]:
    distances = sorted(
        (math.hypot(point[0] - target[0], point[1] - target[1]), index)
        for index, point in enumerate(nodes)
    )
    return [index for distance, index in distances if distance <= float(max_distance)][: max(1, int(limit))]


def _support_graph_shortest_path(
    nodes: list[tuple[float, float]],
    edges: list[tuple[int, int, float]],
    *,
    start_index: int,
    goal_indices: set[int],
) -> tuple[list[int], float]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for source, target, distance in edges:
        if 0 <= source < len(nodes) and 0 <= target < len(nodes) and distance > 1e-6:
            adjacency[source].append((target, float(distance)))
    queue: list[tuple[float, int]] = [(0.0, int(start_index))]
    costs: dict[int, float] = {int(start_index): 0.0}
    previous: dict[int, int] = {}
    reached_goal: int | None = None
    while queue:
        cost, current = heapq.heappop(queue)
        if cost > costs.get(current, float("inf")) + 1e-9:
            continue
        if current in goal_indices and current != int(start_index):
            reached_goal = current
            break
        for target, edge_cost in adjacency[current]:
            next_cost = cost + edge_cost
            if next_cost + 1e-9 >= costs.get(target, float("inf")):
                continue
            costs[target] = next_cost
            previous[target] = current
            heapq.heappush(queue, (next_cost, target))
    if reached_goal is None:
        return [], float("inf")
    path = [reached_goal]
    while path[-1] != int(start_index):
        parent = previous.get(path[-1])
        if parent is None:
            return [], float("inf")
        path.append(parent)
    path.reverse()
    return path, costs.get(reached_goal, float("inf"))


def _support_path_local_subgoal(
    path: list[tuple[float, float]],
    *,
    start: tuple[float, float],
    lookahead_m: float,
    goal: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if len(path) < 2:
        return None
    travelled = 0.0
    previous = start
    for point in path:
        segment = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if segment <= 1e-6:
            previous = point
            continue
        if travelled + segment + 1e-6 >= float(lookahead_m):
            remaining = max(0.0, float(lookahead_m) - travelled)
            ratio = min(1.0, remaining / segment)
            candidate = (
                previous[0] + (point[0] - previous[0]) * ratio,
                previous[1] + (point[1] - previous[1]) * ratio,
            )
            if not polygon or _point_in_polygon(candidate, polygon):
                return candidate
            if not polygon or _point_in_polygon(point, polygon):
                return point
            return None
        travelled += segment
        previous = point
    final = path[-1]
    if math.hypot(final[0] - goal[0], final[1] - goal[1]) <= max(float(lookahead_m), 1.0):
        return goal
    return final if not polygon or _point_in_polygon(final, polygon) else None


def _candidate_toward(start: tuple[float, float], target: tuple[float, float], distance_m: float) -> tuple[float, float]:
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return start
    step = min(float(distance_m), distance)
    return (start[0] + dx / distance * step, start[1] + dy / distance * step)


def _nearest_point_distance(point: tuple[float, float], points: list[tuple[float, float]]) -> float:
    if not points:
        return float("inf")
    return min(math.hypot(point[0] - support[0], point[1] - support[1]) for support in points)


def _stitched_support_route(
    routes: list[list[tuple[float, float]]],
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    max_bridge_m: float,
) -> list[tuple[float, float]]:
    if not routes:
        return []
    remaining = set(range(len(routes)))
    current_index = min(
        remaining,
        key=lambda route_index: min(math.hypot(point[0] - start[0], point[1] - start[1]) for point in routes[route_index]),
    )
    stitched = list(routes[current_index])
    remaining.remove(current_index)
    current_end = stitched[-1]
    current_goal_distance = math.hypot(current_end[0] - goal[0], current_end[1] - goal[1])

    while remaining:
        best: tuple[float, float, int] | None = None
        for route_index in remaining:
            route = routes[route_index]
            bridge_distance = math.hypot(route[0][0] - current_end[0], route[0][1] - current_end[1])
            route_goal_distance = math.hypot(route[-1][0] - goal[0], route[-1][1] - goal[1])
            if bridge_distance > max_bridge_m:
                continue
            if route_goal_distance >= current_goal_distance - SUPPORT_ROUTE_MIN_GOAL_PROGRESS_M:
                continue
            candidate = (bridge_distance, route_goal_distance, route_index)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            break
        _, current_goal_distance, current_index = best
        next_route = routes[current_index]
        if math.hypot(next_route[0][0] - current_end[0], next_route[0][1] - current_end[1]) <= 1e-6:
            stitched.extend(next_route[1:])
        else:
            stitched.extend(next_route)
        current_end = stitched[-1]
        remaining.remove(current_index)

    return stitched if len(stitched) >= 2 else []


def _model_support_points(world_model: Any) -> list[tuple[float, float]]:
    metadata = getattr(world_model, "metadata", {})
    if not isinstance(metadata, dict):
        return []
    route = _coerce_route(metadata.get("support_points", []))
    if route:
        return _deduplicate_points(route)
    points: list[tuple[float, float]] = []
    raw_routes = metadata.get("support_routes")
    if isinstance(raw_routes, list):
        for raw_route in raw_routes:
            points.extend(_coerce_route(raw_route))
    return _deduplicate_points(points)


def _model_support_routes(world_model: Any) -> list[list[tuple[float, float]]]:
    metadata = getattr(world_model, "metadata", {})
    if not isinstance(metadata, dict):
        return []
    raw_routes = metadata.get("support_routes")
    routes: list[list[tuple[float, float]]] = []
    if isinstance(raw_routes, list):
        for raw_route in raw_routes:
            route = _coerce_route(raw_route)
            if len(route) >= 2:
                routes.append(route)
    if routes:
        return routes
    route = _coerce_route(metadata.get("support_points", []))
    return [route] if len(route) >= 2 else []


def _model_support_graph(world_model: Any) -> dict[str, Any]:
    metadata = getattr(world_model, "metadata", {})
    if not isinstance(metadata, dict):
        return {"nodes": [], "edges": []}
    raw_graph = metadata.get("support_graph")
    if not isinstance(raw_graph, dict):
        return {"nodes": [], "edges": []}
    nodes = _coerce_route(raw_graph.get("nodes", []))
    if len(nodes) < 2:
        return {"nodes": [], "edges": []}
    edges: list[tuple[int, int, float]] = []
    for raw_edge in raw_graph.get("edges", []) or []:
        if not isinstance(raw_edge, dict):
            continue
        try:
            source = int(raw_edge.get("source"))
            target = int(raw_edge.get("target"))
        except (TypeError, ValueError):
            continue
        if not (0 <= source < len(nodes) and 0 <= target < len(nodes)) or source == target:
            continue
        try:
            distance = float(raw_edge.get("distance_m"))
        except (TypeError, ValueError):
            distance = math.hypot(nodes[target][0] - nodes[source][0], nodes[target][1] - nodes[source][1])
        if math.isfinite(distance) and distance > 1e-6:
            edges.append((source, target, distance))
    return {"nodes": nodes, "edges": edges}


def _deduplicate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduplicated: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in points:
        key = (round(float(x), 3), round(float(y), 3))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((float(x), float(y)))
    return deduplicated


def _coerce_route(raw: Any) -> list[tuple[float, float]]:
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
