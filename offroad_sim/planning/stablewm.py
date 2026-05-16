"""Planner bridge for stable-worldmodel / LE-WM cost checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning.types import ActionPlanner, PlanningResult
from offroad_sim.utils.runtime_env import prepare_stable_worldmodel_runtime
from offroad_sim.world_models import BaseWorldModel


class StableWorldModelUnavailableError(RuntimeError):
    """Raised when stable-worldmodel planning dependencies are missing."""


class LeWMCEMPlanner(ActionPlanner):
    """CEM planner that uses stable-worldmodel `AutoCostModel` checkpoints."""

    planner_type = "le_wm_cem"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        horizon: int = 10,
        num_samples: int = 256,
        iterations: int = 8,
        topk: int = 32,
        device: str = "cpu",
        seed: int = 0,
        image_size: int = 64,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.horizon = max(1, int(horizon))
        self.num_samples = max(4, int(num_samples))
        self.iterations = max(1, int(iterations))
        self.topk = max(1, min(int(topk), self.num_samples))
        self.device = device
        self.seed = int(seed)
        self.image_size = max(16, int(image_size))
        self._loaded_checkpoint: str | None = None
        self._solver: Any | None = None
        self._torch: Any | None = None

    @classmethod
    def runtime_status(cls) -> dict[str, Any]:
        import importlib.util

        return {
            "stable_worldmodel_available": importlib.util.find_spec("stable_worldmodel") is not None,
            "torch_available": importlib.util.find_spec("torch") is not None,
            "gymnasium_available": importlib.util.find_spec("gymnasium") is not None,
        }

    def plan(
        self,
        observation: Observation,
        world_model: BaseWorldModel,
        *,
        reference_action: Action | None = None,
    ) -> PlanningResult:
        checkpoint = self.checkpoint_path or getattr(world_model, "checkpoint_path", None)
        if checkpoint is None:
            raise StableWorldModelUnavailableError("LE-WM planning requires a checkpoint path.")

        solver, torch = self._solver_for_checkpoint(checkpoint)
        info = self._observation_to_info(observation, torch)
        outputs = solver(info)
        action_array = np.asarray(outputs["actions"][0], dtype=np.float32)
        actions = [Action(steer=float(row[0]), throttle=float(row[1]), brake=float(row[2])) for row in action_array]
        costs = [float(value) for value in outputs.get("costs", [])]
        states = self._kinematic_preview(observation, actions)
        return PlanningResult(
            actions=actions,
            predicted_states=states,
            costs=costs,
            best_cost=float(costs[0]) if costs else 0.0,
            metadata={
                "planner": self.planner_type,
                "checkpoint_path": str(checkpoint),
                "horizon": self.horizon,
                "num_samples": self.num_samples,
                "iterations": self.iterations,
                "topk": self.topk,
            },
        )

    def _solver_for_checkpoint(self, checkpoint: str | Path) -> tuple[Any, Any]:
        checkpoint_key = str(checkpoint)
        if self._solver is not None and self._torch is not None and self._loaded_checkpoint == checkpoint_key:
            return self._solver, self._torch

        prepare_stable_worldmodel_runtime()
        try:
            import torch
            from gymnasium.spaces import Box
            from stable_worldmodel.policy import AutoCostModel, PlanConfig
            from stable_worldmodel.solver.cem import CEMSolver
        except ImportError as exc:
            raise StableWorldModelUnavailableError(
                "Install stable-worldmodel, torch, and gymnasium before using le_wm_cem planning."
            ) from exc

        cost_model = AutoCostModel(checkpoint_key, cache_dir=self.cache_dir)
        solver = CEMSolver(
            cost_model,
            batch_size=1,
            num_samples=self.num_samples,
            n_steps=self.iterations,
            topk=self.topk,
            device=self.device,
            seed=self.seed,
        )
        action_space = Box(
            low=np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32),
            high=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            dtype=np.float32,
        )
        config = PlanConfig(horizon=self.horizon, receding_horizon=1, history_len=1, action_block=1)
        solver.configure(action_space=action_space, n_envs=1, config=config)
        self._solver = solver
        self._torch = torch
        self._loaded_checkpoint = checkpoint_key
        return solver, torch

    def _observation_to_info(self, observation: Observation, torch: Any) -> dict[str, Any]:
        pixels = self._image_from_observation(observation)
        goal = self._goal_image(observation)
        state = np.asarray(
            [[[observation.vehicle_state.x, observation.vehicle_state.y, observation.vehicle_state.yaw, observation.vehicle_state.speed]]],
            dtype=np.float32,
        )
        goal_state = np.asarray([[[observation.goal[0], observation.goal[1]]]], dtype=np.float32)
        return {
            "pixels": torch.from_numpy(pixels[None, None, ...]).float(),
            "goal": torch.from_numpy(goal[None, None, ...]).float(),
            "goal_state": torch.from_numpy(goal_state).float(),
            "state": torch.from_numpy(state).float(),
        }

    def _image_from_observation(self, observation: Observation) -> np.ndarray:
        if observation.front_rgb is not None:
            image = np.asarray(observation.front_rgb)
            if image.ndim == 3 and image.shape[-1] >= 3:
                return _resize_nearest(image[..., :3], self.image_size).astype(np.float32)
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        _draw_marker(canvas, 0.5, 0.5, (255, 255, 255))
        return canvas

    def _goal_image(self, observation: Observation) -> np.ndarray:
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        state = observation.vehicle_state
        dx = observation.goal[0] - state.x
        dy = observation.goal[1] - state.y
        scale = max(abs(dx), abs(dy), 1.0)
        _draw_marker(canvas, 0.5 + 0.45 * dx / scale, 0.5 + 0.45 * dy / scale, (0, 255, 0))
        _draw_marker(canvas, 0.5, 0.5, (255, 255, 255))
        return canvas

    def _kinematic_preview(self, observation: Observation, actions: list[Action]) -> list[VehicleState]:
        states: list[VehicleState] = []
        x = observation.vehicle_state.x
        y = observation.vehicle_state.y
        yaw = observation.vehicle_state.yaw
        speed = observation.vehicle_state.speed
        for action in actions:
            speed = max(0.0, speed + 0.2 * action.throttle - 0.4 * action.brake)
            yaw += 0.08 * action.steer
            x += speed * np.cos(yaw) * 0.1
            y += speed * np.sin(yaw) * 0.1
            states.append(VehicleState(x=float(x), y=float(y), z=observation.vehicle_state.z, yaw=float(yaw), speed=float(speed)))
        return states


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    rows = np.linspace(0, height - 1, size).astype(int)
    cols = np.linspace(0, width - 1, size).astype(int)
    return image[rows][:, cols]


def _draw_marker(canvas: np.ndarray, x_norm: float, y_norm: float, color: tuple[int, int, int]) -> None:
    x = int(np.clip(x_norm, 0.0, 1.0) * (canvas.shape[1] - 1))
    y = int(np.clip(y_norm, 0.0, 1.0) * (canvas.shape[0] - 1))
    radius = max(1, canvas.shape[0] // 24)
    canvas[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = color
