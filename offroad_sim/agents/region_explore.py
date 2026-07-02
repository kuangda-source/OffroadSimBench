"""Region-bounded exploration agent for self-supervised BeamNG collection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from offroad_sim.agents.base import OffroadAgent
from offroad_sim.agents.basic import RuleBasedGoalAgent
from offroad_sim.core import Action, Observation


class RegionExploreAgent(OffroadAgent):
    """Pick temporary goals inside the navigation polygon and drive toward them."""

    def __init__(
        self,
        seed: int | None = None,
        waypoint_radius_m: float = 7.0,
        max_target_steps: int = 80,
        cruise_throttle: float = 0.45,
        goal_bias_interval: int = 4,
        goal_corridor_interval: int = 2,
        goal_corridor_lateral_m: float = 2.0,
        coverage_grid_size: int = 0,
        coverage_target_interval: int = 0,
        route_target_interval: int = 0,
        route_lateral_m: float = 0.0,
        **_: Any,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.waypoint_radius_m = float(waypoint_radius_m)
        self.max_target_steps = max(1, int(max_target_steps))
        self.goal_bias_interval = max(0, int(goal_bias_interval))
        self.goal_corridor_interval = max(0, int(goal_corridor_interval))
        self.goal_corridor_lateral_m = max(0.0, float(goal_corridor_lateral_m))
        self.coverage_grid_size = max(0, int(coverage_grid_size))
        self.coverage_target_interval = max(0, int(coverage_target_interval))
        self.route_target_interval = max(0, int(route_target_interval))
        self.route_lateral_m = max(0.0, float(route_lateral_m))
        self.driver = RuleBasedGoalAgent(cruise_throttle=cruise_throttle)
        self._target: tuple[float, float] | None = None
        self._target_steps = 0
        self._target_count = 0
        self._target_source = "none"
        self._coverage_targets: list[tuple[float, float]] = []
        self._coverage_index = 0
        self._coverage_signature: tuple[tuple[float, float], ...] = ()
        self._route_targets: list[tuple[float, float]] = []
        self._route_index = 0
        self._route_signature: tuple[tuple[float, float], ...] = ()
        self._last_diagnostics: dict[str, Any] = {}

    def reset(self, scenario_info: Any) -> None:
        self._target = None
        self._target_steps = 0
        self._target_count = 0
        self._target_source = "none"
        self._coverage_targets = []
        self._coverage_index = 0
        self._coverage_signature = ()
        self._route_targets = []
        self._route_index = 0
        self._route_signature = ()
        self._last_diagnostics = {}
        self.driver.reset(scenario_info)

    def act(self, obs: Observation) -> Action:
        polygon = _navigation_polygon(obs.info)
        if not polygon:
            action = self.driver.act(obs)
            self._last_diagnostics = {"agent": "region_explorer", "target_in_region": False, "fallback": "goal"}
            return action

        state = obs.vehicle_state
        current = (float(state.x), float(state.y))
        target_reached = self._target is not None and _distance(current, self._target) <= self.waypoint_radius_m
        target_timed_out = self._target_steps >= self.max_target_steps
        can_timeout_target = self._target_source != "route"
        if self._target is None or target_reached or (target_timed_out and can_timeout_target):
            self._target, self._target_source = self._next_target(obs, polygon)
            self._target_steps = 0
            self._target_count += 1
        self._target_steps += 1
        routed = Observation(
            timestamp=obs.timestamp,
            vehicle_state=obs.vehicle_state,
            goal=self._target,
            front_rgb=obs.front_rgb,
            depth=obs.depth,
            lidar_points=obs.lidar_points,
            local_bev=obs.local_bev,
            terrain_map=obs.terrain_map,
            info=obs.info,
        )
        action = self.driver.act(routed)
        self._last_diagnostics = {
            "agent": "region_explorer",
            "target": [float(self._target[0]), float(self._target[1])],
            "target_source": self._target_source,
            "target_in_region": _point_in_polygon(self._target, polygon),
            "target_steps": self._target_steps,
            "target_count": self._target_count,
            "coverage_target_count": len(self._coverage_targets),
            "route_target_count": len(self._route_targets),
        }
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        return None

    def _next_target(self, obs: Observation, polygon: list[tuple[float, float]]) -> tuple[tuple[float, float], str]:
        route_target = self._next_route_target(obs, polygon)
        if route_target is not None:
            return route_target, "route"
        coverage_target = self._next_coverage_target(polygon)
        if coverage_target is not None:
            return coverage_target, "coverage"
        goal = (float(obs.goal[0]), float(obs.goal[1]))
        if (
            self.goal_bias_interval > 0
            and self._target_count % self.goal_bias_interval == 0
            and _point_in_polygon(goal, polygon)
        ):
            return goal, "goal"
        corridor_target = self._goal_corridor_target(obs, polygon)
        if corridor_target is not None:
            return corridor_target, "goal_corridor"
        return _sample_point_in_polygon(polygon, self.rng), "sampled"

    def _next_route_target(self, obs: Observation, polygon: list[tuple[float, float]]) -> tuple[float, float] | None:
        if self.route_target_interval <= 0 or self._target_count % self.route_target_interval != 0:
            return None
        route = _navigation_route(obs.info)
        if len(route) < 2:
            return None
        signature = tuple((round(float(x), 3), round(float(y), 3)) for x, y in route)
        if signature != self._route_signature:
            self._route_signature = signature
            start = (float(obs.vehicle_state.x), float(obs.vehicle_state.y))
            self._route_targets = [
                point
                for point in route[1:]
                if _distance(start, point) > max(0.5, self.waypoint_radius_m * 0.5) and _point_in_polygon(point, polygon)
            ]
            if not self._route_targets:
                self._route_targets = [point for point in route[1:] if _point_in_polygon(point, polygon)]
            self._route_index = 0
        if not self._route_targets:
            return None
        base = self._route_targets[min(self._route_index, len(self._route_targets) - 1)]
        perturbed = self._route_perturbation(base, polygon)
        self._route_index = min(self._route_index + 1, len(self._route_targets) - 1)
        return perturbed

    def _route_perturbation(self, point: tuple[float, float], polygon: list[tuple[float, float]]) -> tuple[float, float]:
        if self.route_lateral_m <= 0.0 or len(self._route_targets) < 2:
            return point
        index = min(self._route_index, len(self._route_targets) - 1)
        previous_point = self._route_targets[index - 1] if index > 0 else self._route_targets[index]
        next_point = self._route_targets[index + 1] if index + 1 < len(self._route_targets) else self._route_targets[index]
        dx = float(next_point[0]) - float(previous_point[0])
        dy = float(next_point[1]) - float(previous_point[1])
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return point
        lateral = float(self.rng.uniform(-self.route_lateral_m, self.route_lateral_m))
        candidate = (float(point[0]) - dy / length * lateral, float(point[1]) + dx / length * lateral)
        return candidate if _point_in_polygon(candidate, polygon) else point

    def _goal_corridor_target(self, obs: Observation, polygon: list[tuple[float, float]]) -> tuple[float, float] | None:
        if self.goal_corridor_interval <= 0 or self._target_count % self.goal_corridor_interval != 0:
            return None
        state = obs.vehicle_state
        start = (float(state.x), float(state.y))
        goal = (float(obs.goal[0]), float(obs.goal[1]))
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= max(self.waypoint_radius_m, 1.0) or not _point_in_polygon(goal, polygon):
            return None
        along = float(self.rng.uniform(0.35, 0.8))
        lateral = float(self.rng.uniform(-self.goal_corridor_lateral_m, self.goal_corridor_lateral_m))
        ux = dx / distance
        uy = dy / distance
        point = (start[0] + dx * along - uy * lateral, start[1] + dy * along + ux * lateral)
        if _point_in_polygon(point, polygon):
            return point
        midpoint = (start[0] + dx * 0.5, start[1] + dy * 0.5)
        if _point_in_polygon(midpoint, polygon):
            return midpoint
        return None

    def _next_coverage_target(self, polygon: list[tuple[float, float]]) -> tuple[float, float] | None:
        if self.coverage_grid_size <= 1 or self.coverage_target_interval <= 0:
            return None
        if self._target_count % self.coverage_target_interval != 0:
            return None
        signature = tuple((round(float(x), 3), round(float(y), 3)) for x, y in polygon)
        if signature != self._coverage_signature:
            self._coverage_signature = signature
            targets = _coverage_points_in_polygon(polygon, self.coverage_grid_size)
            order = self.rng.permutation(len(targets)) if len(targets) > 1 else range(len(targets))
            self._coverage_targets = [targets[int(index)] for index in order]
            self._coverage_index = 0
        if not self._coverage_targets:
            return None
        target = self._coverage_targets[self._coverage_index % len(self._coverage_targets)]
        self._coverage_index += 1
        return target


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


def _navigation_route(info: dict[str, Any]) -> list[tuple[float, float]]:
    raw: Any = []
    if isinstance(info, dict):
        if isinstance(info.get("route"), list):
            raw = info.get("route")
        else:
            navigation = info.get("navigation_region", {})
            if isinstance(navigation, dict):
                for key in ("route", "expert_route"):
                    if isinstance(navigation.get(key), list):
                        raw = navigation.get(key)
                        break
    route: list[tuple[float, float]] = []
    for point in raw or []:
        try:
            route.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return route


def _sample_point_in_polygon(polygon: list[tuple[float, float]], rng: np.random.Generator) -> tuple[float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    for _ in range(256):
        point = (float(rng.uniform(min(xs), max(xs))), float(rng.uniform(min(ys), max(ys))))
        if _point_in_polygon(point, polygon):
            return point
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


def _coverage_points_in_polygon(polygon: list[tuple[float, float]], grid_size: int) -> list[tuple[float, float]]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cols = max(2, int(grid_size))
    rows = max(2, int(grid_size))
    step_x = (max_x - min_x) / cols
    step_y = (max_y - min_y) / rows
    targets: list[tuple[float, float]] = []
    for row in range(rows):
        for col in range(cols):
            point = (min_x + (col + 0.5) * step_x, min_y + (row + 0.5) * step_y)
            if _point_in_polygon(point, polygon):
                targets.append((float(point[0]), float(point[1])))
    if not targets:
        targets.append((float(sum(xs) / len(xs)), float(sum(ys) / len(ys))))
    return targets


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


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
