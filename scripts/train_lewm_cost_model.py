"""Train a small stable-worldmodel cost checkpoint from exported HDF5 data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.utils.runtime_env import prepare_stable_worldmodel_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_hdf5")
    parser.add_argument("--output", default="outputs/models/lewm_cost_smoke")
    parser.add_argument("--checkpoint-name", default="lewm_cost")
    args = parser.parse_args()

    prepare_stable_worldmodel_runtime()
    import h5py  # type: ignore
    import torch

    from offroad_sim.planning.lewm_cost_model import LeWMCostModel, LeWMCostModelConfig

    with h5py.File(args.input_hdf5, "r") as h5:
        states = np.asarray(h5["state"], dtype=np.float32)
        actions = np.asarray(h5["action"], dtype=np.float32)
        goals = np.asarray(h5["goal"], dtype=np.float32) if "goal" in h5 else np.zeros((len(states), 2), dtype=np.float32)
        ep_len = np.asarray(h5["ep_len"], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"], dtype=np.int64)

    config = _fit_config(states, actions, ep_len, ep_offset)
    metadata: dict[str, Any] = {
        "source_hdf5": str(Path(args.input_hdf5).resolve()),
        "sample_count": int(len(states)),
        "episode_count": int(len(ep_len)),
        "mean_goal_distance": float(np.mean(np.linalg.norm(states[:, :2] - goals[:, :2], axis=1))) if len(states) else math.nan,
        "ep_len": ep_len.astype(int).tolist(),
        "ep_offset": ep_offset.astype(int).tolist(),
        "model_kind": "stable_worldmodel_cost_smoke",
    }
    model = LeWMCostModel(config=config, metadata=metadata).eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"{args.checkpoint_name}_object.ckpt"
    torch.save(model, checkpoint_path)
    summary = {
        "output_dir": str(output.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "stable_worldmodel_run": str(output.resolve()),
        "config": model.to_metadata()["config"],
        "metadata": metadata,
    }
    (output / "metadata.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _fit_config(states: np.ndarray, actions: np.ndarray, ep_len: np.ndarray, ep_offset: np.ndarray) -> Any:
    from offroad_sim.planning.lewm_cost_model import LeWMCostModelConfig

    transition_indexes = _transition_indexes(len(states), ep_len, ep_offset)
    if len(transition_indexes) == 0:
        return LeWMCostModelConfig()
    current = states[transition_indexes]
    nxt = states[transition_indexes + 1]
    action = actions[transition_indexes]
    delta_xy = nxt[:, :2] - current[:, :2]
    delta_speed = nxt[:, 4] - current[:, 4]
    delta_yaw = np.asarray([_wrap_angle(float(value)) for value in (nxt[:, 3] - current[:, 3])], dtype=np.float32)
    speed = np.maximum(current[:, 4], 1e-3)
    distance = np.linalg.norm(delta_xy, axis=1)
    dt = _safe_median(distance / speed, default=0.1, low=0.02, high=2.0)
    steer_gain = _safe_median(delta_yaw[np.abs(action[:, 0]) > 1e-3] / action[np.abs(action[:, 0]) > 1e-3, 0], default=0.08, low=0.01, high=0.5)
    throttle_gain = _safe_median(delta_speed[action[:, 1] > 1e-3] / action[action[:, 1] > 1e-3, 1], default=0.2, low=0.02, high=2.0)
    brake_gain = _safe_median(-delta_speed[action[:, 2] > 1e-3] / action[action[:, 2] > 1e-3, 2], default=0.4, low=0.02, high=3.0)
    return LeWMCostModelConfig(
        dt=dt,
        steer_gain=abs(steer_gain),
        throttle_gain=abs(throttle_gain),
        brake_gain=abs(brake_gain),
    )


def _transition_indexes(sample_count: int, ep_len: np.ndarray, ep_offset: np.ndarray) -> np.ndarray:
    indexes: list[np.ndarray] = []
    if len(ep_len) != len(ep_offset):
        return np.arange(max(0, sample_count - 1), dtype=np.int64)
    for length, offset in zip(ep_len.astype(np.int64), ep_offset.astype(np.int64), strict=False):
        start = max(0, int(offset))
        stop = min(sample_count, start + max(0, int(length)))
        if stop - start > 1:
            indexes.append(np.arange(start, stop - 1, dtype=np.int64))
    if not indexes:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(indexes)


def _safe_median(values: np.ndarray, *, default: float, low: float, high: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return default
    return float(np.clip(np.median(finite), low, high))


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


if __name__ == "__main__":
    raise SystemExit(main())
