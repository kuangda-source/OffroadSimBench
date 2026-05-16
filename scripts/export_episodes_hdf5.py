"""Export recorded OffroadSimBench episodes to stable-worldmodel HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_root", help="One episode directory or a directory containing episode directories.")
    parser.add_argument("output_hdf5")
    args = parser.parse_args()

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required for HDF5 export. Install .[lewm] first.") from exc

    episodes = _episode_dirs(Path(args.episode_root))
    if not episodes:
        raise SystemExit("No recorded episodes found.")

    ep_lengths: list[int] = []
    ep_offsets: list[int] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    metadata: dict[str, Any] = {}
    offset = 0
    for episode in episodes:
        rows = _read_episode_steps(episode)
        if not rows:
            continue
        ep_lengths.append(len(rows))
        ep_offsets.append(offset)
        state = np.asarray([row["state"] for row in rows], dtype=np.float32)
        goal = np.asarray([row["goal"] for row in rows], dtype=np.float32)
        states.append(state)
        actions.append(np.asarray([row["action"] for row in rows], dtype=np.float32))
        rewards.append(np.asarray([row["reward"] for row in rows], dtype=np.float32))
        goals.append(goal)
        timestamps.append(np.asarray([row["timestamp"] for row in rows], dtype=np.float32))
        metadata[episode.name] = _read_json(episode / "metadata.json")
        offset += len(rows)

    if not states:
        raise SystemExit("No valid episode steps found.")

    output = Path(args.output_hdf5)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.attrs["schema"] = "stable_worldmodel_flat_v1"
        h5.attrs["source_episode_root"] = str(Path(args.episode_root).resolve())
        h5.attrs["episode_metadata"] = json.dumps(metadata, default=str)
        h5.create_dataset("ep_len", data=np.asarray(ep_lengths, dtype=np.int64))
        h5.create_dataset("ep_offset", data=np.asarray(ep_offsets, dtype=np.int64))
        h5.create_dataset("state", data=np.concatenate(states, axis=0), compression="gzip")
        h5.create_dataset("action", data=np.concatenate(actions, axis=0), compression="gzip")
        h5.create_dataset("reward", data=np.concatenate(rewards, axis=0), compression="gzip")
        h5.create_dataset("goal", data=np.concatenate(goals, axis=0), compression="gzip")
        h5.create_dataset("timestamp", data=np.concatenate(timestamps, axis=0), compression="gzip")

    print(json.dumps({"output_hdf5": str(output.resolve()), "episode_count": len(ep_lengths), "total_frames": offset}, indent=2))
    return 0


def _episode_dirs(root: Path) -> list[Path]:
    if (root / "steps.jsonl").exists():
        return [root]
    return [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "steps.jsonl").exists()]


def _read_episode_steps(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (path / "steps.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            observation = record.get("observation", {})
            vehicle = observation.get("vehicle_state", {}) if isinstance(observation, dict) else {}
            action = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            goal = _goal_pair(observation.get("goal") if isinstance(observation, dict) else None, vehicle)
            rows.append(
                {
                    "state": [
                        _float(vehicle.get("x")),
                        _float(vehicle.get("y")),
                        _float(vehicle.get("z")),
                        _float(vehicle.get("yaw")),
                        _float(vehicle.get("speed")),
                    ],
                    "action": [_float(action.get("steer")), _float(action.get("throttle")), _float(action.get("brake"))],
                    "reward": _float(record.get("reward")),
                    "goal": goal,
                    "timestamp": _float(observation.get("timestamp") if isinstance(observation, dict) else None),
                }
            )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _goal_pair(value: Any, vehicle: dict[str, Any]) -> list[float]:
    fallback = [_float(vehicle.get("x")), _float(vehicle.get("y"))]
    if isinstance(value, dict):
        return [_float(value.get("x", value.get("lon", fallback[0]))), _float(value.get("y", value.get("lat", fallback[1])))]
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=object).reshape(-1)
        return [_float(values[0] if len(values) > 0 else fallback[0]), _float(values[1] if len(values) > 1 else fallback[1])]
    return fallback


if __name__ == "__main__":
    raise SystemExit(main())
