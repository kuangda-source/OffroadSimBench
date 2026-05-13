"""Lightweight 2.5D heightmap backend for quick off-road experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from offroad_sim.backends.base import OffroadSimBackend
from offroad_sim.core.types import Action, Observation, StepResult, VehicleState
from offroad_sim.scenarios import ScenarioConfig


@dataclass(slots=True)
class HeightmapWorld:
    heightmap: np.ndarray
    occupancy_map: np.ndarray
    traversability_map: np.ndarray
    terrain_risk_map: np.ndarray


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class GymHeightmapBackend(OffroadSimBackend):
    """Small deterministic backend with simple kinematics and terrain risk."""

    def __init__(
        self,
        map_size: tuple[int, int] = (128, 128),
        cell_size_m: float = 1.0,
        dt: float = 0.1,
        seed: int = 7,
    ) -> None:
        self.map_size = map_size
        self.cell_size_m = cell_size_m
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.world: HeightmapWorld | None = None
        self.scenario: ScenarioConfig | None = None
        self.vehicle_state = VehicleState()
        self.goal = (0.0, 0.0)
        self.timestamp = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        self.path_length = 0.0
        self.collision_count = 0
        self.success = False
        self.terrain_risk_sum = 0.0
        self.last_distance_to_goal = 0.0

    def reset(self, scenario_config: ScenarioConfig | dict[str, Any]) -> Observation:
        self.scenario = self._coerce_scenario(scenario_config)
        self.goal = self.scenario.task.goal
        self.timestamp = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        self.path_length = 0.0
        self.collision_count = 0
        self.success = False
        self.terrain_risk_sum = 0.0
        self.world = self._generate_world(self.scenario)

        start_x, start_y = self.scenario.task.start
        yaw = math.atan2(self.goal[1] - start_y, self.goal[0] - start_x)
        self.vehicle_state = VehicleState(x=start_x, y=start_y, yaw=yaw, speed=0.0)
        self._update_pitch_roll()
        self.last_distance_to_goal = self._distance_to_goal()
        return self.get_observation()

    def step(self, action: Action) -> StepResult:
        if self.world is None or self.scenario is None:
            raise RuntimeError("Call reset() before step().")

        action = Action(
            steer=_clip(float(action.steer), -1.0, 1.0),
            throttle=_clip(float(action.throttle), 0.0, 1.0),
            brake=_clip(float(action.brake), 0.0, 1.0),
        )

        prev_x = self.vehicle_state.x
        prev_y = self.vehicle_state.y
        prev_distance = self._distance_to_goal()

        acceleration = 4.0 * action.throttle - 7.0 * action.brake - 0.18 * self.vehicle_state.speed
        speed = _clip(self.vehicle_state.speed + acceleration * self.dt, 0.0, 15.0)
        steer_rad = math.radians(35.0) * action.steer
        yaw_rate = 0.0 if abs(steer_rad) < 1e-6 else speed / 2.8 * math.tan(steer_rad)
        yaw = self.vehicle_state.yaw + yaw_rate * self.dt

        x = self.vehicle_state.x + speed * math.cos(yaw) * self.dt
        y = self.vehicle_state.y + speed * math.sin(yaw) * self.dt
        self.vehicle_state = VehicleState(x=x, y=y, yaw=yaw, speed=speed)
        self._update_pitch_roll()

        self.timestamp += self.dt
        self.step_count += 1
        self.path_length += _distance((prev_x, prev_y), (x, y))

        collision = self._is_collision(x, y)
        terrain_risk = self._terrain_risk_at(x, y)
        self.terrain_risk_sum += terrain_risk

        distance_to_goal = self._distance_to_goal()
        progress_reward = (prev_distance - distance_to_goal) * 2.0
        reward = progress_reward - 0.01 - terrain_risk * 0.35

        if collision:
            self.collision_count += 1
            reward -= 10.0
            self.vehicle_state.speed = 0.0

        success_radius = self.scenario.task.success_radius_m
        self.success = distance_to_goal <= success_radius
        if self.success:
            reward += 25.0

        too_far = distance_to_goal > max(200.0, self.last_distance_to_goal * 2.5 + 50.0)
        timed_out = self.timestamp >= self.scenario.task.max_time_sec
        terminated = self.success or collision or too_far
        truncated = timed_out and not terminated
        self.total_reward += reward
        self.last_distance_to_goal = distance_to_goal

        obs = self.get_observation()
        info = {
            "collision": collision,
            "success": self.success,
            "terrain_risk": terrain_risk,
            "distance_to_goal": distance_to_goal,
            "path_length": self.path_length,
            "too_far": too_far,
        }
        obs.info.update(info)
        return StepResult(
            observation=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_observation(self) -> Observation:
        if self.world is None:
            raise RuntimeError("Call reset() before get_observation().")

        terrain_risk = self._terrain_risk_at(self.vehicle_state.x, self.vehicle_state.y)
        collision = self._is_collision(self.vehicle_state.x, self.vehicle_state.y)
        return Observation(
            timestamp=self.timestamp,
            vehicle_state=self.vehicle_state,
            goal=self.goal,
            local_bev=self._local_bev(self.vehicle_state.x, self.vehicle_state.y),
            terrain_map=np.stack(
                [
                    self.world.heightmap,
                    self.world.traversability_map,
                    self.world.terrain_risk_map,
                    self.world.occupancy_map.astype(float),
                ],
                axis=0,
            ),
            info={
                "terrain_risk": terrain_risk,
                "collision": collision,
                "distance_to_goal": self._distance_to_goal(),
            },
        )

    def get_metrics(self) -> dict[str, Any]:
        average_risk = self.terrain_risk_sum / self.step_count if self.step_count else 0.0
        average_speed = self.path_length / self.timestamp if self.timestamp > 0 else 0.0
        return {
            "success": self.success,
            "total_reward": float(self.total_reward),
            "episode_length": self.step_count,
            "elapsed_time_sec": float(self.timestamp),
            "path_length": float(self.path_length),
            "average_speed": float(average_speed),
            "collision_count": self.collision_count,
            "average_terrain_risk": float(average_risk),
            "distance_to_goal": float(self._distance_to_goal()) if self.world is not None else 0.0,
        }

    def close(self) -> None:
        return None

    def _coerce_scenario(self, scenario_config: ScenarioConfig | dict[str, Any]) -> ScenarioConfig:
        if isinstance(scenario_config, ScenarioConfig):
            return scenario_config
        return ScenarioConfig.from_dict(scenario_config)

    def _generate_world(self, scenario: ScenarioConfig) -> HeightmapWorld:
        width, height = self.map_size
        x = np.linspace(0.0, 1.0, width)
        y = np.linspace(0.0, 1.0, height)
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")

        difficulty_scale = {"easy": 0.75, "medium": 1.0, "hard": 1.35}.get(
            scenario.terrain.difficulty,
            1.0,
        )
        base = (
            0.35 * np.sin(grid_x * math.tau * 2.0)
            + 0.22 * np.cos(grid_y * math.tau * 2.7)
            + 0.16 * np.sin((grid_x + grid_y) * math.tau * 1.4)
        )
        noise = self.rng.normal(0.0, 0.04 * difficulty_scale, size=(height, width))
        heightmap = (base + noise) * difficulty_scale

        grad_y, grad_x = np.gradient(heightmap)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        slope_norm = slope / max(float(slope.max()), 1e-6)

        obstacle_probability = min(0.09 * difficulty_scale, 0.18)
        random_obstacles = self.rng.random((height, width)) < obstacle_probability

        start = scenario.task.start
        goal = scenario.task.goal
        trail_mask = self._trail_mask(start, goal, width, height)
        occupancy_map = random_obstacles & ~trail_mask
        risk = np.clip(0.25 * difficulty_scale + 0.65 * slope_norm + 0.25 * occupancy_map, 0.0, 1.0)
        risk[trail_mask] *= 0.35
        traversability = np.clip(1.0 - risk, 0.0, 1.0)

        self._clear_disk(occupancy_map, start, radius_m=4.0)
        self._clear_disk(occupancy_map, goal, radius_m=5.0)
        return HeightmapWorld(
            heightmap=heightmap.astype(np.float32),
            occupancy_map=occupancy_map,
            traversability_map=traversability.astype(np.float32),
            terrain_risk_map=risk.astype(np.float32),
        )

    def _trail_mask(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        yy, xx = np.indices((height, width))
        sx, sy = start[0] / self.cell_size_m, start[1] / self.cell_size_m
        gx, gy = goal[0] / self.cell_size_m, goal[1] / self.cell_size_m
        vx, vy = gx - sx, gy - sy
        length_sq = max(vx * vx + vy * vy, 1e-6)
        projection = np.clip(((xx - sx) * vx + (yy - sy) * vy) / length_sq, 0.0, 1.0)
        closest_x = sx + projection * vx
        closest_y = sy + projection * vy
        distance = np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)
        return distance <= 5.0

    def _clear_disk(self, occupancy_map: np.ndarray, center: tuple[float, float], radius_m: float) -> None:
        cx, cy = center[0] / self.cell_size_m, center[1] / self.cell_size_m
        yy, xx = np.indices(occupancy_map.shape)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius_m / self.cell_size_m) ** 2
        occupancy_map[mask] = False

    def _index_at(self, x: float, y: float) -> tuple[int, int] | None:
        if self.world is None:
            return None
        col = int(round(x / self.cell_size_m))
        row = int(round(y / self.cell_size_m))
        height, width = self.world.heightmap.shape
        if row < 0 or col < 0 or row >= height or col >= width:
            return None
        return row, col

    def _terrain_risk_at(self, x: float, y: float) -> float:
        if self.world is None:
            return 0.0
        index = self._index_at(x, y)
        if index is None:
            return 1.0
        row, col = index
        return float(self.world.terrain_risk_map[row, col])

    def _is_collision(self, x: float, y: float) -> bool:
        if self.world is None:
            return False
        index = self._index_at(x, y)
        if index is None:
            return True
        row, col = index
        return bool(self.world.occupancy_map[row, col])

    def _local_bev(self, x: float, y: float, radius_cells: int = 12) -> np.ndarray:
        if self.world is None:
            raise RuntimeError("World is not initialized.")

        index = self._index_at(x, y)
        if index is None:
            row = int(round(y / self.cell_size_m))
            col = int(round(x / self.cell_size_m))
        else:
            row, col = index

        arrays = [
            self.world.heightmap,
            self.world.traversability_map,
            self.world.terrain_risk_map,
            self.world.occupancy_map.astype(np.float32),
        ]
        padded = [np.pad(array, radius_cells, mode="edge") for array in arrays]
        row += radius_cells
        col += radius_cells
        patch = [
            array[
                row - radius_cells : row + radius_cells + 1,
                col - radius_cells : col + radius_cells + 1,
            ]
            for array in padded
        ]
        return np.stack(patch, axis=0).astype(np.float32)

    def _distance_to_goal(self) -> float:
        return _distance((self.vehicle_state.x, self.vehicle_state.y), self.goal)

    def _update_pitch_roll(self) -> None:
        if self.world is None:
            return
        index = self._index_at(self.vehicle_state.x, self.vehicle_state.y)
        if index is None:
            self.vehicle_state.pitch = 0.0
            self.vehicle_state.roll = 0.0
            return
        row, col = index
        heightmap = self.world.heightmap
        row0 = max(0, row - 1)
        row1 = min(heightmap.shape[0] - 1, row + 1)
        col0 = max(0, col - 1)
        col1 = min(heightmap.shape[1] - 1, col + 1)
        dx = float(heightmap[row, col1] - heightmap[row, col0]) / max((col1 - col0) * self.cell_size_m, 1e-6)
        dy = float(heightmap[row1, col] - heightmap[row0, col]) / max((row1 - row0) * self.cell_size_m, 1e-6)
        self.vehicle_state.pitch = math.atan(dy)
        self.vehicle_state.roll = math.atan(dx)

