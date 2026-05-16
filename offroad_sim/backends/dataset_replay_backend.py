"""Dataset replay backend built on the dataset adapter registry."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from offroad_sim.backends.base import OffroadSimBackend
from offroad_sim.core import Action, Observation, StepResult
from offroad_sim.datasets import DatasetRegistry, DatasetSequence, default_dataset_registry


class DatasetReplayBackend(OffroadSimBackend):
    """Replay recorded dataset frames through the normal backend interface."""

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        sequence_id: str | None = None,
        adapter: str | None = None,
        load_assets: bool = False,
        max_steps: int | None = None,
        registry: DatasetRegistry | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.sequence_id = sequence_id
        self.adapter_name = adapter
        self.load_assets = load_assets
        self.max_steps = max_steps
        self.registry = registry or default_dataset_registry()

        self._sequence: DatasetSequence | None = None
        self._adapter_used: str | None = None
        self._index = 0
        self._step_count = 0

    def reset(self, scenario_config: Any = None) -> Observation:
        root = self._resolve_root(scenario_config)
        sequence_id = self._read_config(scenario_config, "sequence_id", self.sequence_id)
        adapter_name = self._read_config(scenario_config, "adapter", self.adapter_name)
        self.load_assets = bool(self._read_config(scenario_config, "load_assets", self.load_assets))

        adapter = self.registry.resolve(root, adapter_name)
        if sequence_id is None:
            sequence_ids = adapter.list_sequences(root)
            if not sequence_ids:
                raise ValueError(f"Dataset has no sequences: {root}")
            sequence_id = sequence_ids[0]

        self._sequence = adapter.load_sequence(root, sequence_id)
        self._adapter_used = adapter.name
        self._index = 0
        self._step_count = 0
        return self.get_observation()

    def step(self, action: Action) -> StepResult:
        sequence = self._require_sequence()
        if self._index < len(sequence.frames) - 1:
            self._index += 1

        self._step_count += 1
        terminated = self._index >= len(sequence.frames) - 1
        truncated = bool(self.max_steps is not None and self._step_count >= self.max_steps and not terminated)
        observation = self.get_observation()
        info = {
            "backend": "dataset_replay",
            "adapter": self._adapter_used,
            "dataset_id": sequence.dataset_id,
            "sequence_id": sequence.sequence_id,
            "frame_id": observation.info["frame_id"],
            "frame_index": self._index,
            "action_ignored": True,
            "input_action": {"steer": action.steer, "throttle": action.throttle, "brake": action.brake},
        }
        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_observation(self) -> Observation:
        sequence = self._require_sequence()
        frame = sequence.frames[self._index]
        assets = frame.available_assets()
        info = {
            "backend": "dataset_replay",
            "adapter": self._adapter_used,
            "dataset_id": sequence.dataset_id,
            "dataset_type": sequence.dataset_type,
            "sequence_id": sequence.sequence_id,
            "frame_id": frame.frame_id,
            "frame_index": self._index,
            "frame_count": len(sequence.frames),
            "assets": assets,
            "frame_metadata": frame.metadata,
        }
        return Observation(
            timestamp=frame.timestamp,
            vehicle_state=frame.vehicle_state,
            goal=sequence.goal or (frame.vehicle_state.x, frame.vehicle_state.y),
            front_rgb=self._load_npy_asset(frame.front_rgb_path),
            depth=self._load_npy_asset(frame.depth_path),
            lidar_points=self._load_npy_asset(frame.lidar_path),
            local_bev=self._load_npy_asset(frame.local_bev_path),
            terrain_map=self._load_npy_asset(frame.terrain_map_path),
            info=info,
        )

    def get_metrics(self) -> dict[str, Any]:
        if self._sequence is None:
            return {"backend": "dataset_replay", "reset": False}

        done = self._index >= len(self._sequence.frames) - 1
        return {
            "backend": "dataset_replay",
            "adapter": self._adapter_used,
            "dataset_id": self._sequence.dataset_id,
            "dataset_type": self._sequence.dataset_type,
            "sequence_id": self._sequence.sequence_id,
            "frames_total": len(self._sequence.frames),
            "frames_played": self._index + 1,
            "episode_length": self._step_count,
            "current_index": self._index,
            "done": done,
        }

    def close(self) -> None:
        self._sequence = None
        self._adapter_used = None
        self._index = 0
        self._step_count = 0

    def _resolve_root(self, scenario_config: Any) -> Path:
        root = self._read_config(scenario_config, "dataset_root", self.dataset_root)
        if root is None:
            raise ValueError("DatasetReplayBackend requires dataset_root in init or reset config.")
        return Path(root)

    def _read_config(self, config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, Mapping):
            return config.get(key, default)
        return getattr(config, key, default)

    def _require_sequence(self) -> DatasetSequence:
        if self._sequence is None:
            raise RuntimeError("DatasetReplayBackend has not been reset.")
        return self._sequence

    def _load_npy_asset(self, path: str | None) -> Any:
        if not self.load_assets or path is None:
            return None
        if path.startswith("zip://"):
            member_suffix = Path(path.rsplit("!", 1)[-1]).suffix
            if member_suffix != ".npy":
                return path
            zip_path, member = _split_zip_uri(path)
            with zipfile.ZipFile(zip_path) as archive:
                return np.load(io.BytesIO(archive.read(member)))
        asset_path = Path(path)
        if asset_path.suffix != ".npy":
            return path
        return np.load(asset_path)


def _split_zip_uri(path: str) -> tuple[str, str]:
    raw = path.removeprefix("zip://")
    zip_path, member = raw.split("!", 1)
    return zip_path, member
