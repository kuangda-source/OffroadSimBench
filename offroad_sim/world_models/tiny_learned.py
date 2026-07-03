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


FEATURE_NAMES = ("bias", "speed", "yaw_sin", "yaw_cos", "steer", "throttle", "brake", "gear")
OUTPUT_NAMES = ("dx", "dy", "dyaw", "dspeed")
MAX_SUPPORT_POINTS = 512
DEFAULT_SUPPORT_RADIUS_M = 8.0


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
        raw_weights = np.asarray(weights, dtype=np.float64) if weights is not None else np.zeros(default_shape)
        self.weights = _normalize_weights(raw_weights, default_shape)
        if self.weights.shape != default_shape:
            raise ValueError(f"Expected weights shape {default_shape}, got {self.weights.shape}")
        self.dt = float(dt)
        self.ridge = float(ridge)
        self.metadata = dict(metadata or {})
        self.metadata["feature_names"] = list(FEATURE_NAMES)
        self.metadata["output_names"] = list(OUTPUT_NAMES)
        self.support_points = _support_points_array(self.metadata.get("support_points"))
        self.support_radius_m = max(1.0, float(self.metadata.get("support_radius_m", DEFAULT_SUPPORT_RADIUS_M) or DEFAULT_SUPPORT_RADIUS_M))

    @classmethod
    def fit(cls, sequences: Iterable[DatasetSequence], *, ridge: float = 1e-4, validation_fraction: float = 0.2) -> "TinyLearnedWorldModel":
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        segment_labels: list[str] = []
        frame_count = 0
        sequence_count = 0
        recorded_action_sample_count = 0
        support_points: list[tuple[float, float]] = []
        support_routes: list[list[list[float]]] = []

        for sequence in sequences:
            sequence_count += 1
            frames = sequence.frames
            frame_count += len(frames)
            sequence_support_points: list[tuple[float, float]] = []
            for frame in frames:
                point = (float(frame.vehicle_state.x), float(frame.vehicle_state.y))
                support_points.append(point)
                sequence_support_points.append(point)
            route = _downsample_support_points(sequence_support_points, max_points=MAX_SUPPORT_POINTS)
            if len(route) >= 2:
                support_routes.append(route)
            for transition_index, (current, nxt) in enumerate(zip(frames, frames[1:])):
                action, action_source = _transition_action(current, nxt)
                if action_source == "recorded":
                    recorded_action_sample_count += 1
                features.append(_features(current.vehicle_state, action))
                targets.append(_target(current.vehicle_state, nxt.vehicle_state))
                segment_labels.append(_transition_segment(sequence, transition_index, current.vehicle_state))

        if not features:
            raise ValueError("TinyLearnedWorldModel.fit requires at least two frames.")

        x = np.vstack(features)
        y = np.vstack(targets)
        train_indices, validation_indices = _train_validation_indices(int(x.shape[0]), validation_fraction=validation_fraction)
        train_x = x[train_indices]
        train_y = y[train_indices]
        regularizer = ridge * np.eye(x.shape[1])
        weights = np.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ train_y)
        train_residual = train_x @ weights - train_y
        validation_residual = x[validation_indices] @ weights - y[validation_indices] if validation_indices.size else np.empty((0, y.shape[1]))
        all_residual = x @ weights - y
        segment_rmse, segment_sample_count = _segment_error_summary(all_residual, segment_labels)
        metadata = {
            "sequence_count": sequence_count,
            "frame_count": frame_count,
            "sample_count": int(x.shape[0]),
            "transition_count": int(x.shape[0]),
            "train_sample_count": int(train_indices.size),
            "validation_sample_count": int(validation_indices.size),
            "validation_fraction": float(max(0.0, min(0.9, validation_fraction))),
            "train_mse": float(np.mean(train_residual**2)),
            "train_rmse": float(np.sqrt(np.mean(train_residual**2))),
            "validation_mse": float(np.mean(validation_residual**2)) if validation_indices.size else math.nan,
            "validation_rmse": float(np.sqrt(np.mean(validation_residual**2))) if validation_indices.size else math.nan,
            "segment_rmse": segment_rmse,
            "segment_sample_count": segment_sample_count,
            "recorded_action_sample_count": recorded_action_sample_count,
            "feature_names": list(FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
            "support_points": _downsample_support_points(support_points, max_points=MAX_SUPPORT_POINTS),
            "support_routes": support_routes,
            "support_route_count": len(support_routes),
            "support_point_count": len(support_points),
            "support_radius_m": DEFAULT_SUPPORT_RADIUS_M,
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
        support_distance, support_risk = _support_distance_and_risk(states, self.support_points, self.support_radius_m)
        if math.isfinite(support_risk) and support_risk > 0.0:
            risk_samples.append(float(support_risk))
        metadata: dict[str, Any] = {
            "model_type": self.model_type,
            "horizon": horizon,
            "mean_risk": float(np.mean(risk_samples)) if risk_samples else 0.0,
            "max_risk": float(np.max(risk_samples)) if risk_samples else 0.0,
            "train_rmse": self.metadata.get("train_rmse"),
            "support_distance_m": float(support_distance),
            "support_risk": float(support_risk),
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
            float(action.gear) if action.gear is not None else 1.0,
        ],
        dtype=np.float64,
    )


def _downsample_support_points(points: list[tuple[float, float]], *, max_points: int) -> list[list[float]]:
    if not points:
        return []
    limit = max(1, int(max_points))
    if len(points) <= limit:
        selected = points
    else:
        indices = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
        selected = [points[int(index)] for index in indices]
    deduped: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in selected:
        key = (round(float(x), 3), round(float(y), 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([float(x), float(y)])
    return deduped


def _support_points_array(raw: Any) -> np.ndarray:
    if not isinstance(raw, list):
        return np.empty((0, 2), dtype=np.float64)
    points: list[tuple[float, float]] = []
    for point in raw:
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _support_distance_and_risk(states: list[VehicleState], support_points: np.ndarray, support_radius_m: float) -> tuple[float, float]:
    if support_points.size == 0 or not states:
        return math.nan, 0.0
    predicted = np.asarray([[float(state.x), float(state.y)] for state in states], dtype=np.float64)
    deltas = predicted[:, None, :] - support_points[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    support_distance = float(np.min(distances))
    radius = max(1.0, float(support_radius_m))
    support_risk = max(0.0, support_distance - radius) / radius
    return support_distance, support_risk


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
        gear=1,
    )


def _transition_action(current: DatasetFrame, nxt: DatasetFrame) -> tuple[Action, str]:
    # Episode records store the command that produced a frame, so the next
    # frame's action is the best label for the current -> next transition.
    if nxt.action is not None:
        return nxt.action, "recorded"
    if current.action is not None:
        return current.action, "recorded"
    return _action_from_pair(current.vehicle_state, nxt.vehicle_state), "inferred"


def _train_validation_indices(sample_count: int, *, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 1:
        return np.arange(sample_count, dtype=np.int64), np.asarray([], dtype=np.int64)
    fraction = max(0.0, min(0.9, float(validation_fraction)))
    validation_count = int(round(sample_count * fraction))
    if fraction > 0.0:
        validation_count = max(1, validation_count)
    validation_count = min(sample_count - 1, validation_count)
    if validation_count <= 0:
        return np.arange(sample_count, dtype=np.int64), np.asarray([], dtype=np.int64)
    # Deterministic spread over the trajectory keeps start/middle/goal samples represented.
    validation_indices = np.unique(np.linspace(0, sample_count - 1, validation_count, dtype=np.int64))
    if validation_indices.size < validation_count:
        missing = validation_count - int(validation_indices.size)
        candidates = [index for index in range(sample_count - 1, -1, -1) if index not in set(int(item) for item in validation_indices)]
        validation_indices = np.asarray(sorted([*validation_indices.tolist(), *candidates[:missing]]), dtype=np.int64)
    validation_set = set(int(index) for index in validation_indices)
    train_indices = np.asarray([index for index in range(sample_count) if index not in validation_set], dtype=np.int64)
    return train_indices, validation_indices


def _transition_segment(sequence: DatasetSequence, transition_index: int, state: VehicleState) -> str:
    if sequence.goal is not None and sequence.frames:
        start_xy = _sequence_task_start(sequence) or (
            float(sequence.frames[0].vehicle_state.x),
            float(sequence.frames[0].vehicle_state.y),
        )
        start_distance = math.hypot(float(start_xy[0]) - float(sequence.goal[0]), float(start_xy[1]) - float(sequence.goal[1]))
        current_distance = math.hypot(float(state.x) - float(sequence.goal[0]), float(state.y) - float(sequence.goal[1]))
        progress = 0.0 if start_distance <= 1e-9 else max(0.0, min(1.0, (start_distance - current_distance) / start_distance))
    else:
        transition_count = max(1, len(sequence.frames) - 1)
        progress = max(0.0, min(1.0, transition_index / transition_count))
    if progress < 1.0 / 3.0:
        return "start"
    if progress < 2.0 / 3.0:
        return "middle"
    return "goal"


def _sequence_task_start(sequence: DatasetSequence) -> tuple[float, float] | None:
    raw = sequence.metadata.get("task_start_pos") if isinstance(sequence.metadata, dict) else None
    if raw is None:
        return None
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError, IndexError):
        return None


def _segment_error_summary(residual: np.ndarray, labels: list[str]) -> tuple[dict[str, float | None], dict[str, int]]:
    rmse: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for name in ("start", "middle", "goal"):
        indices = [index for index, label in enumerate(labels) if label == name]
        counts[name] = len(indices)
        if not indices:
            rmse[name] = None
            continue
        segment_residual = residual[np.asarray(indices, dtype=np.int64)]
        rmse[name] = float(np.sqrt(np.mean(segment_residual**2)))
    return rmse, counts


def _normalize_weights(weights: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    if weights.shape == expected_shape:
        return weights
    legacy_shape = (expected_shape[0] - 1, expected_shape[1])
    if weights.shape == legacy_shape:
        gear_row = np.zeros((1, expected_shape[1]), dtype=np.float64)
        return np.vstack([weights, gear_row])
    return weights
