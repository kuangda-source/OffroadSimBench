"""Analytic kinematic world model baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from offroad_sim.core.types import Action, Observation, VehicleState
from offroad_sim.world_models.base import BaseWorldModel, WorldModelPrediction


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _as_action_sequence(action: Action | Sequence[Action], horizon: int) -> list[Action]:
    if isinstance(action, Action):
        return [action for _ in range(horizon)]
    actions = list(action)
    if not actions:
        return [Action() for _ in range(horizon)]
    if len(actions) >= horizon:
        return actions[:horizon]
    return actions + [actions[-1] for _ in range(horizon - len(actions))]


class SimpleKinematicWorldModel(BaseWorldModel):
    """Deterministic bicycle-model rollout for first-stage planning tests."""

    model_type = "simple_kinematic"

    def __init__(
        self,
        dt: float = 0.1,
        wheelbase_m: float = 2.8,
        max_steer_deg: float = 35.0,
        max_speed_mps: float = 15.0,
    ) -> None:
        self.dt = dt
        self.wheelbase_m = wheelbase_m
        self.max_steer_deg = max_steer_deg
        self.max_speed_mps = max_speed_mps

    def predict(
        self,
        observation: Observation,
        action: Action | Sequence[Action],
        horizon: int = 10,
    ) -> WorldModelPrediction:
        horizon = max(1, int(horizon))
        actions = _as_action_sequence(action, horizon)
        state = observation.vehicle_state
        predicted: list[VehicleState] = []

        x = float(state.x)
        y = float(state.y)
        z = float(state.z)
        yaw = float(state.yaw)
        pitch = float(state.pitch)
        roll = float(state.roll)
        speed = float(state.speed)

        for command in actions:
            steer = _clip(float(command.steer), -1.0, 1.0)
            throttle = _clip(float(command.throttle), 0.0, 1.0)
            brake = _clip(float(command.brake), 0.0, 1.0)

            acceleration = 4.0 * throttle - 7.0 * brake - 0.18 * speed
            speed = _clip(speed + acceleration * self.dt, 0.0, self.max_speed_mps)
            steer_rad = math.radians(self.max_steer_deg) * steer
            yaw_rate = 0.0 if abs(steer_rad) < 1e-6 else speed / self.wheelbase_m * math.tan(steer_rad)
            yaw += yaw_rate * self.dt
            x += speed * math.cos(yaw) * self.dt
            y += speed * math.sin(yaw) * self.dt

            predicted.append(
                VehicleState(
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    speed=speed,
                )
            )

        risk_map, risk_samples = self._extract_risk(observation, predicted)
        metadata: dict[str, Any] = {
            "dt": self.dt,
            "horizon": horizon,
            "mean_risk": float(np.mean(risk_samples)) if risk_samples else 0.0,
            "max_risk": float(np.max(risk_samples)) if risk_samples else 0.0,
            "risk_samples": risk_samples,
        }
        return WorldModelPrediction(
            states=predicted,
            actions=actions,
            risk_map=risk_map,
            metadata=metadata,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "wheelbase_m": self.wheelbase_m,
            "max_steer_deg": self.max_steer_deg,
            "max_speed_mps": self.max_speed_mps,
        }

    @classmethod
    def load(cls, path: str | Path) -> "SimpleKinematicWorldModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model type: {payload.get('model_type')}")
        return cls(**payload.get("config", {}))

    def _extract_risk(
        self,
        observation: Observation,
        states: list[VehicleState],
    ) -> tuple[np.ndarray | None, list[float]]:
        if observation.terrain_map is not None:
            terrain = np.asarray(observation.terrain_map)
            if terrain.ndim == 3 and terrain.shape[0] >= 3:
                risk_layer = terrain[2].astype(np.float32)
                height, width = risk_layer.shape
                samples = []
                for state in states:
                    row = int(round(state.y))
                    col = int(round(state.x))
                    if row < 0 or col < 0 or row >= height or col >= width:
                        samples.append(1.0)
                    else:
                        samples.append(float(risk_layer[row, col]))
                return risk_layer, samples

        if observation.local_bev is not None:
            bev = np.asarray(observation.local_bev)
            if bev.ndim == 3 and bev.shape[0] >= 3:
                risk_layer = bev[2].astype(np.float32)
                return risk_layer, [float(np.mean(risk_layer)), float(np.max(risk_layer))]

        return None, []
