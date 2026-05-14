"""Gymnasium environment wrapper around the local heightmap backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from offroad_sim.backends import GymHeightmapBackend
from offroad_sim.core.types import Action, Observation
from offroad_sim.scenarios import ScenarioConfig, load_scenario_config


DEFAULT_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / "forest_trail_001.yaml"


class OffroadGymEnv(gym.Env):
    """Gymnasium-compatible wrapper for quick RL integration tests."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioConfig | str | Path | None = None,
        seed: int = 7,
        max_episode_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.scenario = self._load_scenario(scenario)
        self.seed_value = seed
        self.max_episode_steps = max_episode_steps
        self.backend = GymHeightmapBackend(seed=seed)
        self._elapsed_steps = 0

        height, width = self.backend.map_size[1], self.backend.map_size[0]
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "local_bev": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 25, 25), dtype=np.float32),
                "terrain_map": spaces.Box(low=-np.inf, high=np.inf, shape=(4, height, width), dtype=np.float32),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)
            self.backend = GymHeightmapBackend(seed=self.seed_value)
        self._elapsed_steps = 0
        observation = self.backend.reset(self.scenario)
        return self._to_gym_observation(observation), self._info(observation)

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, float, float],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float32).reshape(3)
        result = self.backend.step(
            Action(
                steer=float(command[0]),
                throttle=float(command[1]),
                brake=float(command[2]),
            )
        )
        self._elapsed_steps += 1
        truncated = bool(result.truncated)
        if self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps:
            truncated = truncated or not result.terminated
        info = dict(result.info)
        info["elapsed_steps"] = self._elapsed_steps
        return (
            self._to_gym_observation(result.observation),
            float(result.reward),
            bool(result.terminated),
            truncated,
            info,
        )

    def close(self) -> None:
        self.backend.close()

    def _to_gym_observation(self, observation: Observation) -> dict[str, np.ndarray]:
        state = observation.vehicle_state
        return {
            "state": np.asarray(
                [state.x, state.y, state.z, state.yaw, state.pitch, state.roll, state.speed],
                dtype=np.float32,
            ),
            "goal": np.asarray(observation.goal, dtype=np.float32),
            "local_bev": np.asarray(observation.local_bev, dtype=np.float32),
            "terrain_map": np.asarray(observation.terrain_map, dtype=np.float32),
        }

    def _info(self, observation: Observation) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "backend": "gym_heightmap",
            **dict(observation.info),
        }

    def _load_scenario(self, scenario: ScenarioConfig | str | Path | None) -> ScenarioConfig:
        if isinstance(scenario, ScenarioConfig):
            return scenario
        return load_scenario_config(scenario or DEFAULT_SCENARIO_PATH)


def make_gymnasium_env(**kwargs: Any) -> OffroadGymEnv:
    return OffroadGymEnv(**kwargs)
