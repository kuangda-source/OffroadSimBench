"""Region and boundary cost helpers for navigation tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from offroad_sim.core import VehicleState


Point2 = tuple[float, float]


@dataclass(slots=True)
class RegionTrajectoryCost:
    polygon: list[Point2]
    out_of_region_weight: float = 250.0
    boundary_weight: float = 8.0
    boundary_margin_m: float = 3.0

    @classmethod
    def from_task(cls, task: Mapping[str, Any] | None) -> "RegionTrajectoryCost | None":
        if not isinstance(task, Mapping):
            return None
        region = task.get("region", {})
        polygon = region.get("polygon", []) if isinstance(region, Mapping) else []
        points = [_point2(point) for point in polygon or []]
        if len(points) < 3:
            return None
        cost = task.get("cost", {})
        cost = cost if isinstance(cost, Mapping) else {}
        return cls(
            polygon=points,
            out_of_region_weight=float(cost.get("out_of_region_weight", 250.0)),
            boundary_weight=float(cost.get("boundary_weight", 8.0)),
            boundary_margin_m=float(cost.get("boundary_margin_m", 3.0)),
        )

    def evaluate(self, states: Iterable[VehicleState]) -> float:
        total = 0.0
        for state in states:
            point = (float(state.x), float(state.y))
            if not self.contains(point):
                total += self.out_of_region_weight
                continue
            if self.boundary_margin_m > 0.0 and self.boundary_weight > 0.0:
                distance = self.distance_to_boundary(point)
                if distance < self.boundary_margin_m:
                    total += self.boundary_weight * (self.boundary_margin_m - distance) / self.boundary_margin_m
        return total

    def contains(self, point: Point2) -> bool:
        x, y = point
        inside = False
        vertices = self.polygon
        j = len(vertices) - 1
        for i, current in enumerate(vertices):
            previous = vertices[j]
            if ((current[1] > y) != (previous[1] > y)) and (
                x < (previous[0] - current[0]) * (y - current[1]) / ((previous[1] - current[1]) or 1e-12) + current[0]
            ):
                inside = not inside
            j = i
        return inside

    def distance_to_boundary(self, point: Point2) -> float:
        if len(self.polygon) < 2:
            return math.inf
        return min(_distance_point_to_segment(point, start, end) for start, end in zip(self.polygon, [*self.polygon[1:], self.polygon[0]], strict=False))


def navigation_region_from_observation_info(info: Mapping[str, Any] | None) -> RegionTrajectoryCost | None:
    if not isinstance(info, Mapping):
        return None
    raw = info.get("navigation_region") or info.get("task")
    return RegionTrajectoryCost.from_task(raw) if isinstance(raw, Mapping) else None


def _point2(value: Any) -> Point2:
    items = list(value or [])
    if len(items) < 2:
        raise ValueError("Expected a two-value point.")
    return (float(items[0]), float(items[1]))


def _distance_point_to_segment(point: Point2, start: Point2, end: Point2) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return math.hypot(px - closest_x, py - closest_y)
