"""Lightweight NumPy MLP dynamics model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.datasets import DatasetSequence
from offroad_sim.world_models.base import BaseWorldModel, WorldModelPrediction
from offroad_sim.world_models.tiny_learned import (
    DEFAULT_SUPPORT_RADIUS_M,
    FEATURE_NAMES,
    MAX_SUPPORT_POINTS,
    OUTPUT_NAMES,
    _as_action_sequence,
    _build_support_graph,
    _downsample_support_points,
    _features,
    _segment_error_summary,
    _support_distance_and_risk,
    _support_points_array,
    _target,
    _train_validation_indices,
    _transition_action,
    _transition_segment,
    _wrap_angle,
)


class MLPDynamicsWorldModel(BaseWorldModel):
    """A tiny random-feature MLP dynamics model with ridge-trained readout."""

    model_type = "mlp_dynamics"

    def __init__(
        self,
        hidden_weights: np.ndarray | None = None,
        hidden_bias: np.ndarray | None = None,
        output_weights: np.ndarray | None = None,
        *,
        hidden_size: int = 32,
        seed: int = 13,
        ridge: float = 1e-4,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.hidden_size = max(1, int(hidden_size))
        self.seed = int(seed)
        self.ridge = float(ridge)
        input_size = len(FEATURE_NAMES)
        output_size = len(OUTPUT_NAMES)
        rng = np.random.default_rng(self.seed)
        self.hidden_weights = (
            np.asarray(hidden_weights, dtype=np.float64)
            if hidden_weights is not None
            else rng.normal(0.0, 1.0 / math.sqrt(input_size), size=(input_size, self.hidden_size))
        )
        self.hidden_bias = (
            np.asarray(hidden_bias, dtype=np.float64)
            if hidden_bias is not None
            else rng.normal(0.0, 0.1, size=(self.hidden_size,))
        )
        self.output_weights = (
            np.asarray(output_weights, dtype=np.float64)
            if output_weights is not None
            else np.zeros((self.hidden_size + 1, output_size), dtype=np.float64)
        )
        if self.hidden_weights.shape != (input_size, self.hidden_size):
            raise ValueError(f"Expected hidden_weights shape {(input_size, self.hidden_size)}, got {self.hidden_weights.shape}")
        if self.hidden_bias.shape != (self.hidden_size,):
            raise ValueError(f"Expected hidden_bias shape {(self.hidden_size,)}, got {self.hidden_bias.shape}")
        if self.output_weights.shape != (self.hidden_size + 1, output_size):
            raise ValueError(f"Expected output_weights shape {(self.hidden_size + 1, output_size)}, got {self.output_weights.shape}")
        self.metadata = dict(metadata or {})
        self.metadata["feature_names"] = list(FEATURE_NAMES)
        self.metadata["output_names"] = list(OUTPUT_NAMES)
        self.metadata["model_family"] = "random_feature_mlp"
        self.metadata["hidden_size"] = self.hidden_size
        self.support_points = _support_points_array(self.metadata.get("support_points"))
        self.support_radius_m = max(1.0, float(self.metadata.get("support_radius_m", DEFAULT_SUPPORT_RADIUS_M) or DEFAULT_SUPPORT_RADIUS_M))

    @classmethod
    def fit(
        cls,
        sequences: Iterable[DatasetSequence],
        *,
        hidden_size: int = 32,
        seed: int = 13,
        ridge: float = 1e-4,
        validation_fraction: float = 0.2,
    ) -> "MLPDynamicsWorldModel":
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
            raise ValueError("MLPDynamicsWorldModel.fit requires at least two frames.")

        x = np.vstack(features)
        y = np.vstack(targets)
        train_indices, validation_indices = _train_validation_indices(int(x.shape[0]), validation_fraction=validation_fraction)
        model = cls(hidden_size=hidden_size, seed=seed, ridge=ridge)
        train_hidden = model._hidden_features(x[train_indices])
        regularizer = ridge * np.eye(train_hidden.shape[1])
        model.output_weights = np.linalg.solve(train_hidden.T @ train_hidden + regularizer, train_hidden.T @ y[train_indices])
        train_residual = train_hidden @ model.output_weights - y[train_indices]
        validation_residual = (
            model._hidden_features(x[validation_indices]) @ model.output_weights - y[validation_indices]
            if validation_indices.size
            else np.empty((0, y.shape[1]))
        )
        all_residual = model._hidden_features(x) @ model.output_weights - y
        segment_rmse, segment_sample_count = _segment_error_summary(all_residual, segment_labels)
        model.metadata.update(
            {
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
                "support_points": _downsample_support_points(support_points, max_points=MAX_SUPPORT_POINTS),
                "support_routes": support_routes,
                "support_graph": _build_support_graph(support_routes),
                "support_route_count": len(support_routes),
                "support_point_count": len(support_points),
                "support_radius_m": DEFAULT_SUPPORT_RADIUS_M,
            }
        )
        model.support_points = _support_points_array(model.metadata.get("support_points"))
        return model

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
            delta = self._hidden_features(_features(current, command)[None, :])[0] @ self.output_weights
            x += float(delta[0])
            y += float(delta[1])
            yaw = _wrap_angle(yaw + float(delta[2]))
            speed = max(0.0, speed + float(delta[3]))
            states.append(VehicleState(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, speed=speed))

        support_distance, support_risk = _support_distance_and_risk(states, self.support_points, self.support_radius_m)
        risk_samples = [float(support_risk)] if math.isfinite(support_risk) and support_risk > 0.0 else []
        metadata = {
            "model_type": self.model_type,
            "horizon": horizon,
            "mean_risk": float(np.mean(risk_samples)) if risk_samples else 0.0,
            "max_risk": float(np.max(risk_samples)) if risk_samples else 0.0,
            "train_rmse": self.metadata.get("train_rmse"),
            "validation_rmse": self.metadata.get("validation_rmse"),
            "support_distance_m": float(support_distance),
            "support_risk": float(support_risk),
        }
        return WorldModelPrediction(states=states, actions=actions, risk_map=None, metadata=metadata)

    def get_config(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "seed": self.seed,
            "ridge": self.ridge,
            "metadata": self.metadata,
            "feature_names": list(FEATURE_NAMES),
            "output_names": list(OUTPUT_NAMES),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output_dir = output.parent if output.suffix else output
        metadata_path = output if output.suffix else output / "model.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "weights.npz"
        np.savez(
            weights_path,
            hidden_weights=self.hidden_weights,
            hidden_bias=self.hidden_bias,
            output_weights=self.output_weights,
        )
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
    def load(cls, path: str | Path) -> "MLPDynamicsWorldModel":
        input_path = Path(path)
        metadata_path = input_path / "model.json" if input_path.is_dir() else input_path
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model type: {payload.get('model_type')}")
        config = dict(payload.get("config", {}))
        arrays = np.load(metadata_path.parent / str(payload.get("weights", "weights.npz")))
        return cls(
            hidden_weights=arrays["hidden_weights"],
            hidden_bias=arrays["hidden_bias"],
            output_weights=arrays["output_weights"],
            hidden_size=int(config.get("hidden_size", arrays["hidden_bias"].shape[0])),
            seed=int(config.get("seed", 13)),
            ridge=float(config.get("ridge", 1e-4)),
            metadata=dict(config.get("metadata", {})),
        )

    def _hidden_features(self, x: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x, dtype=np.float64)
        hidden = np.tanh(x2 @ self.hidden_weights + self.hidden_bias)
        return np.hstack([np.ones((hidden.shape[0], 1), dtype=np.float64), hidden])
