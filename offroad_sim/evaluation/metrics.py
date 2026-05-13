"""Episode metrics tracking for simulator backends and demos."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from offroad_sim.core import Action, Observation, StepResult, VehicleState


def _state_distance(a: VehicleState, b: VehicleState) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


@dataclass(slots=True)
class MetricsTracker:
    """Accumulate common off-road episode metrics."""

    total_reward: float = 0.0
    episode_length: int = 0
    elapsed_time_sec: float = 0.0
    time_to_goal: float | None = None
    path_length: float = 0.0
    speed_sum: float = 0.0
    max_speed: float = 0.0
    collision_count: int = 0
    rollover: bool = False
    max_pitch: float = 0.0
    max_roll: float = 0.0
    terrain_risk_sum: float = 0.0
    terrain_risk_count: int = 0
    control_delta_sum: float = 0.0
    control_delta_count: int = 0
    success: bool = False
    last_action: Action | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.total_reward = 0.0
        self.episode_length = 0
        self.elapsed_time_sec = 0.0
        self.time_to_goal = None
        self.path_length = 0.0
        self.speed_sum = 0.0
        self.max_speed = 0.0
        self.collision_count = 0
        self.rollover = False
        self.max_pitch = 0.0
        self.max_roll = 0.0
        self.terrain_risk_sum = 0.0
        self.terrain_risk_count = 0
        self.control_delta_sum = 0.0
        self.control_delta_count = 0
        self.success = False
        self.last_action = None
        self.extras.clear()

    def update(self, observation: Observation, action: Action, step_result: StepResult) -> None:
        next_obs = step_result.observation
        next_state = next_obs.vehicle_state

        self.episode_length += 1
        self.total_reward += float(step_result.reward)
        self.elapsed_time_sec = max(self.elapsed_time_sec, float(next_obs.timestamp))
        self.path_length += _state_distance(observation.vehicle_state, next_state)
        self.speed_sum += float(next_state.speed)
        self.max_speed = max(self.max_speed, float(next_state.speed))
        self.max_pitch = max(self.max_pitch, abs(float(next_state.pitch)))
        self.max_roll = max(self.max_roll, abs(float(next_state.roll)))

        terrain_risk = step_result.info.get("terrain_risk", next_obs.info.get("terrain_risk"))
        if terrain_risk is not None:
            self.terrain_risk_sum += float(terrain_risk)
            self.terrain_risk_count += 1

        if bool(step_result.info.get("collision", False)):
            self.collision_count += 1

        if bool(step_result.info.get("rollover", False)) or abs(float(next_state.roll)) > math.radians(90):
            self.rollover = True

        if bool(step_result.info.get("success", False)):
            self.success = True
            if self.time_to_goal is None:
                self.time_to_goal = float(next_obs.timestamp)

        if self.last_action is not None:
            delta = (
                abs(float(action.steer) - float(self.last_action.steer))
                + abs(float(action.throttle) - float(self.last_action.throttle))
                + abs(float(action.brake) - float(self.last_action.brake))
            )
            self.control_delta_sum += delta
            self.control_delta_count += 1
        self.last_action = action

    def compute(self) -> dict[str, Any]:
        average_speed = self.speed_sum / self.episode_length if self.episode_length else 0.0
        average_terrain_risk = (
            self.terrain_risk_sum / self.terrain_risk_count if self.terrain_risk_count else 0.0
        )
        control_smoothness = (
            self.control_delta_sum / self.control_delta_count if self.control_delta_count else 0.0
        )
        metrics = {
            "success": self.success,
            "total_reward": float(self.total_reward),
            "episode_length": self.episode_length,
            "elapsed_time_sec": float(self.elapsed_time_sec),
            "time_to_goal": self.time_to_goal,
            "path_length": float(self.path_length),
            "average_speed": float(average_speed),
            "max_speed": float(self.max_speed),
            "collision_count": self.collision_count,
            "rollover": self.rollover,
            "max_pitch": float(self.max_pitch),
            "max_roll": float(self.max_roll),
            "average_terrain_risk": float(average_terrain_risk),
            "control_smoothness": float(control_smoothness),
        }
        metrics.update(self.extras)
        return metrics

