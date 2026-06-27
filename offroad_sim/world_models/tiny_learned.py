"""Small learned dynamics model used for phase-three integration tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.datasets import DatasetSequence
from offroad_sim.world_models.base import BaseWorldModel, WorldModelPrediction


FEATURE_NAMES = ("bias", "speed", "yaw_sin", "yaw_cos", "steer", "throttle", "brake")
OUTPUT_NAMES = ("dx", "dy", "dyaw", "dspeed")


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _as_action_sequence(action: Action | Sequence[Action], horizon: int) -> list[Action]:
    if isinstance(action, Action):
        return [action for _ in range(horizon)]
    actions = list(action)
    if not actions:
        return [Action() for _ in range(horizon)]
    if len(actions) >= horizon:
        return actions[:horizon]
    return actions + [actions[-1] for _ in range(horizon - len(actions))]


class TinyLearnedWorldModel(BaseWorldModel):
    """A NumPy linear dynamics model with a stable registry/load boundary."""

    model_type = "tiny_learned"

    def __init__(
        self,
        weights: np.ndarray | None = None,
        *,
        dt: float = 1.0,
        ridge: float = 1e-4,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        default_shape = (len(FEATURE_NAMES), len(OUTPUT_NAMES))
        self.weights = np.asarray(weights, dtype=np.float64) if weights is not None else np.zeros(default_shape)
        if self.weights.shape != default_shape:
            raise ValueError(f"Expected weights shape {default_shape}, got {self.weights.shape}")
        self.dt = float(dt)
        self.ridge = float(ridge)
        self.metadata = dict(metadata or {})

    @classmethod
    def fit(cls, sequences: Iterable[DatasetSequence], *, ridge: float = 1e-4) -> "TinyLearnedWorldModel":
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        frame_count = 0
        sequence_count = 0
        recorded_action_sample_count = 0

        for sequence in sequences:
            sequence_count += 1
            frames = sequence.frames
            frame_count += len(frames)
            for current, nxt in zip(frames, frames[1:]):
                action, action_source = _transition_action(current, nxt)
                if action_source == "recorded":
                    recorded_action_sample_count += 1
                features.append(_features(current.vehicle_state, action))
                targets.append(_target(current.vehicle_state, nxt.vehicle_state))

        if not features:
            raise ValueError("TinyLearnedWorldModel.fit requires at least two frames.")

        x = np.vstack(features)
        y = np.vstack(targets)
        regularizer = ridge * np.eye(x.shape[1])
        weights = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
        residual = x @ weights - y
        metadata = {
            "sequence_count": sequence_count,
            "frame_count": frame_count,
            "sample_count": int(x.shape[0]),
            "train_mse": float(np.mean(residual**2)),
            "train_rmse": float(np.sqrt(np.mean(residual**2))),
            "recorded_action_sample_count": recorded_action_sample_count,
            "feature_names": list(FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
        }
        return cls(weights=weights, ridge=ridge, metadata=metadata)

    def predict(
        self,
        observation: Observation,
        action: Action | Sequence[Action],
        horizon: int = 10,
    ) -> WorldModelPrediction:
        horizon = max(1, int(horizon))
        actions = _as_action_sequence(action, horizon)
        base = observation.vehicle_state
        x = float(base.x)
        y = float(base.y)
        z = float(base.z)
        yaw = float(base.yaw)
        pitch = float(base.pitch)
        roll = float(base.roll)
        speed = float(base.speed)
        states: list[VehicleState] = []

        for command in actions:
            current = VehicleState(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, speed=speed)
            delta = _features(current, command) @ self.weights
            x += float(delta[0])
            y += float(delta[1])
            yaw = _wrap_angle(yaw + float(delta[2]))
            speed = max(0.0, speed + float(delta[3]))
            states.append(VehicleState(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, speed=speed))

        risk_map, risk_samples = self._risk_from_observation(observation)
        metadata: dict[str, Any] = {
            "model_type": self.model_type,
            "horizon": horizon,
            "mean_risk": float(np.mean(risk_samples)) if risk_samples else 0.0,
            "max_risk": float(np.max(risk_samples)) if risk_samples else 0.0,
            "train_rmse": self.metadata.get("train_rmse"),
        }
        return WorldModelPrediction(states=states, actions=actions, risk_map=risk_map, metadata=metadata)

    def get_config(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "ridge": self.ridge,
            "metadata": self.metadata,
            "feature_names": list(FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        if output.suffix:
            output_dir = output.parent
            metadata_path = output
        else:
            output_dir = output
            metadata_path = output / "model.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "weights.npz"
        np.savez(weights_path, weights=self.weights)
        metadata_path.write_text(
            json.dumps(
                {
                    "model_type": self.model_type,
                    "config": self.get_config(),
                    "weights": weights_path.name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "TinyLearnedWorldModel":
        input_path = Path(path)
        metadata_path = input_path / "model.json" if input_path.is_dir() else input_path
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model type: {payload.get('model_type')}")
        config = dict(payload.get("config", {}))
        weights = np.load(metadata_path.parent / str(payload.get("weights", "weights.npz")))["weights"]
        return cls(
            weights=weights,
            dt=float(config.get("dt", 1.0)),
            ridge=float(config.get("ridge", 1e-4)),
            metadata=dict(config.get("metadata", {})),
        )

    def _risk_from_observation(self, observation: Observation) -> tuple[np.ndarray | None, list[float]]:
        if observation.terrain_map is not None:
            terrain = np.asarray(observation.terrain_map)
            if terrain.ndim == 3 and terrain.shape[0] >= 3:
                risk = terrain[2].astype(np.float32)
                return risk, [float(np.mean(risk)), float(np.max(risk))]
        if observation.local_bev is not None:
            bev = np.asarray(observation.local_bev)
            if bev.ndim == 3 and bev.shape[0] >= 3:
                risk = bev[2].astype(np.float32)
                return risk, [float(np.mean(risk)), float(np.max(risk))]
        return None, []


def _features(state: VehicleState, action: Action) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            float(state.speed),
            math.sin(float(state.yaw)),
            math.cos(float(state.yaw)),
            float(action.steer),
            float(action.throttle),
            float(action.brake),
        ],
        dtype=np.float64,
    )


def _target(current: VehicleState, nxt: VehicleState) -> np.ndarray:
    return np.asarray(
        [
            float(nxt.x) - float(current.x),
            float(nxt.y) - float(current.y),
            _wrap_angle(float(nxt.yaw) - float(current.yaw)),
            float(nxt.speed) - float(current.speed),
        ],
        dtype=np.float64,
    )


def _action_from_pair(current: VehicleState, nxt: VehicleState) -> Action:
    speed_delta = float(nxt.speed) - float(current.speed)
    yaw_delta = _wrap_angle(float(nxt.yaw) - float(current.yaw))
    return Action(
        steer=max(-1.0, min(1.0, yaw_delta / 0.25)),
        throttle=max(0.0, min(1.0, speed_delta / 2.0 + 0.35)),
        brake=max(0.0, min(1.0, -speed_delta / 2.0)),
    )


def _transition_action(current: DatasetFrame, nxt: DatasetFrame) -> tuple[Action, str]:
    # Episode records store the command that produced a frame, so the next
    # frame's action is the best label for the current -> next transition.
    if nxt.action is not None:
        return nxt.action, "recorded"
    if current.action is not None:
        return current.action, "recorded"
    return _action_from_pair(current.vehicle_state, nxt.vehicle_state), "inferred"
