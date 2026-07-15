"""A lightweight RGB-to-depth baseline used to exercise the trainer protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.datasets import DatasetFrame, validate_dataset_split_payload
from offroad_sim.datasets.assets import load_asset_array


def dataset_split_frames(
    adapter: Any,
    dataset_root: str | Path,
    split_path: str | Path,
    split_name: str,
) -> list[DatasetFrame]:
    """Resolve normalized frames selected by a workbench split definition."""

    path = Path(split_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset_split_payload(payload)
    expected_root = str(payload.get("dataset_root") or "").strip()
    if expected_root and Path(expected_root).resolve() != Path(dataset_root).resolve():
        raise ValueError(f"Dataset split belongs to a different dataset root: {expected_root}")
    expected_adapter = str(payload.get("adapter") or "").strip()
    if expected_adapter and expected_adapter != str(adapter.name):
        raise ValueError(f"Dataset split expects adapter '{expected_adapter}', not '{adapter.name}'.")

    splits = payload.get("splits") if isinstance(payload.get("splits"), dict) else {}
    rows = splits.get(split_name)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset split has no samples for '{split_name}'.")

    selected: list[DatasetFrame] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid row in dataset split '{split_name}'.")
        sequence_id = str(row.get("sequence_id") or "").strip()
        if not sequence_id:
            raise ValueError(f"Dataset split '{split_name}' contains a row without sequence_id.")
        sequence = adapter.load_sequence(str(dataset_root), sequence_id)
        raw_indices = row.get("frame_indices")
        if raw_indices is None:
            selected.extend(sequence.frames)
            continue
        if not isinstance(raw_indices, list):
            raise ValueError(f"Dataset split frame_indices must be a list for sequence '{sequence_id}'.")
        for raw_index in raw_indices:
            index = int(raw_index)
            if index < 0 or index >= len(sequence.frames):
                raise ValueError(
                    f"Dataset split frame index {index} is outside sequence '{sequence_id}' "
                    f"with {len(sequence.frames)} frames."
                )
            selected.append(sequence.frames[index])
    if not selected:
        raise ValueError(f"Dataset split has no frames for '{split_name}'.")
    return selected


@dataclass(slots=True)
class TinyDepthModel:
    weights: np.ndarray
    max_depth_m: float
    metadata: dict[str, Any]

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        features = depth_features(rgb)
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Tiny depth model contains non-finite weights.")
        prediction = features @ self.weights
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("Tiny depth inference produced non-finite predictions.")
        return np.clip(prediction.reshape(rgb.shape[:2]), 0.0, self.max_depth_m).astype(np.float32)

    def save(self, output_dir: str | Path) -> Path:
        if not np.all(np.isfinite(self.weights)):
            raise FloatingPointError("Refusing to save a tiny depth model with non-finite weights.")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        weights_path = target / "weights.npz"
        np.savez_compressed(weights_path, weights=self.weights.astype(np.float32))
        model_path = target / "model.json"
        model_path.write_text(
            json.dumps(
                {
                    "model_type": "tiny_rgb_depth",
                    "weights": weights_path.name,
                    "max_depth_m": self.max_depth_m,
                    "metadata": self.metadata,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return model_path

    @classmethod
    def load(cls, artifact: str | Path) -> "TinyDepthModel":
        path = Path(artifact)
        model_path = path / "model.json" if path.is_dir() else path
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        weights_path = model_path.parent / str(payload["weights"])
        weights = np.load(weights_path, allow_pickle=False)["weights"].astype(np.float64)
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"Tiny depth checkpoint contains non-finite weights: {weights_path}")
        return cls(
            weights=weights,
            max_depth_m=float(payload.get("max_depth_m", 80.0)),
            metadata=dict(payload.get("metadata") or {}),
        )


def frame_depth_arrays(frame: DatasetFrame) -> tuple[np.ndarray, np.ndarray]:
    if not frame.front_rgb_path or not frame.depth_path:
        raise ValueError(f"Frame {frame.frame_id} does not contain synchronized RGB and depth assets.")
    rgb = np.asarray(load_asset_array(frame.front_rgb_path))
    depth = np.asarray(load_asset_array(frame.depth_path))
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Unsupported RGB shape for frame {frame.frame_id}: {rgb.shape}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for frame {frame.frame_id}: {rgb.shape} vs {depth.shape}")
    depth_m = depth.astype(np.float32)
    if np.nanpercentile(depth_m, 99) > 250.0:
        depth_m /= 1000.0
    return rgb[..., :3], depth_m


def depth_features(rgb: np.ndarray) -> np.ndarray:
    image = rgb.astype(np.float64)
    if not np.all(np.isfinite(image)):
        raise ValueError("RGB input contains NaN or infinite values.")
    if image.max(initial=0.0) > 1.5:
        image /= 255.0
    height, width = image.shape[:2]
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    x = x_grid.astype(np.float64) / max(width - 1, 1)
    y = y_grid.astype(np.float64) / max(height - 1, 1)
    return np.column_stack(
        [
            np.ones(height * width, dtype=np.float64),
            image[..., 0].reshape(-1),
            image[..., 1].reshape(-1),
            image[..., 2].reshape(-1),
            x.reshape(-1),
            y.reshape(-1),
        ]
    )


def sample_frame_pixels(
    frame: DatasetFrame,
    *,
    max_pixels: int,
    rng: np.random.Generator,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb, depth = frame_depth_arrays(frame)
    features = depth_features(rgb)
    targets = depth.reshape(-1).astype(np.float64)
    valid = np.flatnonzero(
        np.isfinite(targets)
        & np.all(np.isfinite(features), axis=1)
        & (targets > 0.0)
        & (targets <= max_depth_m)
    )
    if valid.size == 0:
        raise ValueError(f"Frame {frame.frame_id} has no valid depth pixels.")
    if valid.size > max_pixels:
        valid = rng.choice(valid, size=max_pixels, replace=False)
    return features[valid], targets[valid]


def depth_colormap(depth: np.ndarray, *, max_depth_m: float) -> np.ndarray:
    normalized = np.clip(np.asarray(depth, dtype=np.float32) / max(max_depth_m, 1e-6), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)
