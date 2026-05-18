"""Navigation-region task contract shared by maps, algorithms, and backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from offroad_sim.utils.yaml_io import load_yaml_file


Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass(slots=True)
class NavigationRegionTask:
    task_id: str
    map_id: str
    level: str
    region_polygon: list[Point2]
    start_pos: Point3
    start_yaw: float
    goal_pos: Point2
    goal_radius: float
    expert_route: list[Point2] = field(default_factory=list)
    backend_targets: list[str] = field(default_factory=lambda: ["beamng"])
    max_steps: int = 300
    max_collision_count: int = 0
    cost: dict[str, Any] = field(default_factory=dict)
    beamng: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NavigationRegionTask":
        if str(data.get("task_type", "navigation_region_v1")) != "navigation_region_v1":
            raise ValueError("Only navigation_region_v1 tasks are supported.")
        region = data.get("region", {})
        start_pose = data.get("start_pose", {})
        goal = data.get("goal", {})
        constraints = data.get("constraints", {})
        return cls(
            task_id=str(data.get("task_id") or data.get("map_id") or "navigation_region"),
            map_id=str(data["map_id"]),
            level=str(data.get("level") or data.get("beamng", {}).get("level") or data["map_id"]),
            region_polygon=_points2(region.get("polygon", [])),
            start_pos=_point3(start_pose.get("pos")),
            start_yaw=float(start_pose.get("yaw", 0.0)),
            goal_pos=_point2(goal.get("pos")),
            goal_radius=float(goal.get("radius", 5.0)),
            expert_route=_points2(data.get("expert_route", [])),
            backend_targets=[str(item) for item in data.get("backend_targets", ["beamng"])],
            max_steps=int(constraints.get("max_steps", 300)),
            max_collision_count=int(constraints.get("max_collision_count", 0)),
            cost=dict(data.get("cost", {})),
            beamng=dict(data.get("beamng", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NavigationRegionTask":
        return cls.from_dict(load_yaml_file(path))

    def contains_point(self, point: Point2) -> bool:
        x, y = point
        inside = False
        vertices = self.region_polygon
        if len(vertices) < 3:
            return False
        j = len(vertices) - 1
        for i, current in enumerate(vertices):
            previous = vertices[j]
            if ((current[1] > y) != (previous[1] > y)) and (
                x < (previous[0] - current[0]) * (y - current[1]) / ((previous[1] - current[1]) or 1e-12) + current[0]
            ):
                inside = not inside
            j = i
        return inside

    def to_beamng_scenario(self, *, mode: str) -> dict[str, Any]:
        if mode not in {"collection", "evaluation"}:
            raise ValueError("mode must be collection or evaluation.")
        if mode == "collection" and not self.expert_route:
            raise ValueError("navigation_region_v1 collection requires expert_route.")
        if mode == "evaluation":
            drive_mode = self.beamng.get("evaluation_drive_mode", "manual")
        else:
            drive_mode = self.beamng.get("collection_drive_mode") or self.beamng.get("drive_mode", "ai_line")
        beamng = {
            "level": self.level,
            "vehicle_model": str(self.beamng.get("vehicle_model", "pickup")),
            "vehicle_start": {
                "pos": list(self.start_pos),
                "rot_quat": list(self.beamng.get("rot_quat", [0.0, 0.0, 0.0, 1.0])),
            },
            "camera_mode": str(self.beamng.get("camera_mode", "orbit")),
            "draw_route": bool(self.beamng.get("draw_route", True)),
            "drive_mode": str(drive_mode),
            "ai_line_speed": float(self.beamng.get("ai_line_speed", 10.0)),
            "steps_per_action": int(self.beamng.get("steps_per_action", 18)),
        }
        if mode == "collection":
            beamng["route"] = [list(point) for point in self.expert_route]
        return {
            "scenario_id": f"{self.task_id}_{mode}",
            "backend": "beamng",
            "map": self.level,
            "weather": str(self.beamng.get("weather", "sunny")),
            "terrain": {"type": "navigation_region", "difficulty": "medium"},
            "task": {
                "max_time_sec": float(self.beamng.get("max_time_sec", 180.0)),
                "success_radius_m": self.goal_radius,
                "start": [self.start_pos[0], self.start_pos[1]],
                "goal": [self.goal_pos[0], self.goal_pos[1]],
            },
            "metrics": {"collision": True, "rollover": True, "path_length": True, "terrain_risk": True},
            "metadata": {
                "task": self.to_dict(),
                "beamng": beamng,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": "navigation_region_v1",
            "map_id": self.map_id,
            "backend_targets": self.backend_targets,
            "level": self.level,
            "region": {"polygon": [list(point) for point in self.region_polygon]},
            "start_pose": {"pos": list(self.start_pos), "yaw": self.start_yaw},
            "goal": {"pos": list(self.goal_pos), "radius": self.goal_radius},
            "expert_route": [list(point) for point in self.expert_route],
            "constraints": {"max_steps": self.max_steps, "max_collision_count": self.max_collision_count},
            "cost": self.cost,
            "beamng": self.beamng,
        }


def load_navigation_region_task(path: str | Path) -> NavigationRegionTask:
    return NavigationRegionTask.from_yaml(path)


def _point2(value: Any) -> Point2:
    values = list(value or [])
    if len(values) < 2:
        raise ValueError("Expected a 2-value point.")
    return (float(values[0]), float(values[1]))


def _point3(value: Any) -> Point3:
    values = list(value or [])
    if len(values) < 3:
        raise ValueError("Expected a 3-value point.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _points2(values: Any) -> list[Point2]:
    return [_point2(value) for value in values or []]
