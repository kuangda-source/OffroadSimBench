"""Episode recording and replay utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from offroad_sim.core import Action, Observation, VehicleState


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _action_to_dict(action: Action) -> dict[str, float]:
    return {
        "steer": float(action.steer),
        "throttle": float(action.throttle),
        "brake": float(action.brake),
    }


def _vehicle_state_to_dict(state: VehicleState) -> dict[str, float]:
    return {
        "x": float(state.x),
        "y": float(state.y),
        "z": float(state.z),
        "yaw": float(state.yaw),
        "pitch": float(state.pitch),
        "roll": float(state.roll),
        "speed": float(state.speed),
    }


def _observation_to_dict(
    observation: Observation,
    arrays_dir: Path | None = None,
    step_index: int = 0,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "timestamp": float(observation.timestamp),
        "vehicle_state": _vehicle_state_to_dict(observation.vehicle_state),
        "goal": [float(observation.goal[0]), float(observation.goal[1])],
        "info": observation.info,
    }

    array_fields = {
        "front_rgb": observation.front_rgb,
        "depth": observation.depth,
        "lidar_points": observation.lidar_points,
        "local_bev": observation.local_bev,
        "terrain_map": observation.terrain_map,
    }
    for field_name, value in array_fields.items():
        if value is None:
            data[field_name] = None
            continue
        if arrays_dir is None:
            data[field_name] = None
            continue
        arrays_dir.mkdir(parents=True, exist_ok=True)
        relative = Path("arrays") / f"step_{step_index:06d}_{field_name}.npy"
        np.save(arrays_dir / relative.name, np.asarray(value))
        data[field_name] = str(relative).replace("\\", "/")

    return data


class EpisodeRecorder:
    """Collect episode steps in memory and save them to disk."""

    def __init__(self, save_arrays: bool = False) -> None:
        self.save_arrays = save_arrays
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self._steps: list[dict[str, Any]] = []

    def start_episode(self, metadata: dict[str, Any]) -> None:
        self.metadata = dict(metadata)
        self.metrics = {}
        self._steps = []

    def record_step(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self._steps.append(
            {
                "observation": observation,
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "info": dict(info),
            }
        )

    def end_episode(self, metrics: dict[str, Any]) -> None:
        self.metrics = dict(metrics)

    def save(self, path: str | Path) -> Path:
        episode_path = Path(path)
        episode_path.mkdir(parents=True, exist_ok=True)
        arrays_dir = episode_path / "arrays" if self.save_arrays else None

        metadata = dict(self.metadata)
        metadata["step_count"] = len(self._steps)
        metadata["format"] = "offroad_sim_episode_v1"

        (episode_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (episode_path / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, default=_json_default),
            encoding="utf-8",
        )

        with (episode_path / "steps.jsonl").open("w", encoding="utf-8") as file:
            for index, step in enumerate(self._steps):
                record = {
                    "step_index": index,
                    "observation": _observation_to_dict(
                        step["observation"],
                        arrays_dir=arrays_dir,
                        step_index=index,
                    ),
                    "action": _action_to_dict(step["action"]),
                    "reward": step["reward"],
                    "done": step["done"],
                    "info": step["info"],
                }
                file.write(json.dumps(record, default=_json_default) + "\n")

        return episode_path


class EpisodePlayer:
    """Load and iterate a saved episode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata = self._load_json("metadata.json")
        self.metrics = self._load_json("metrics.json")

    @classmethod
    def load(cls, path: str | Path) -> "EpisodePlayer":
        return cls(path)

    def iter_steps(self) -> Iterator[dict[str, Any]]:
        steps_path = self.path / "steps.jsonl"
        with steps_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.metrics)

    def _load_json(self, name: str) -> dict[str, Any]:
        file_path = self.path / name
        return json.loads(file_path.read_text(encoding="utf-8"))


def load_episode(path: str | Path) -> EpisodePlayer:
    return EpisodePlayer.load(path)

