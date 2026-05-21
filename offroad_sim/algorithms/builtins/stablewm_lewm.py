"""Adapter for existing stable-worldmodel / upstream LE-WM checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.algorithms.base import AlgorithmAdapter, ScoreActionsRequest, ScoreActionsResult
from offroad_sim.algorithms.manifest import AlgorithmManifest
from offroad_sim.algorithms.registry import AlgorithmStatus
from offroad_sim.planning.lewm_checkpoint import LeWMCheckpointReference, normalize_lewm_checkpoint_reference
from offroad_sim.utils.runtime_env import prepare_stable_worldmodel_runtime


def builtin_manifest() -> AlgorithmManifest:
    return AlgorithmManifest.from_dict(
        {
            "algorithm_id": "stablewm_lewm",
            "display_name": "StableWM / Upstream LE-WM Checkpoint",
            "entrypoint": "offroad_sim.algorithms.builtins.stablewm_lewm:StableWMLeWMAlgorithm",
            "version": "0.1.0",
            "capabilities": {
                "infer": True,
                "score_actions": True,
            },
            "input_contract": {
                "observations": ["state", "goal", "rgb_optional"],
                "actions": ["steer", "throttle", "brake"],
                "task": "navigation_region_v1",
            },
            "output_contract": {"mode": "action_cost"},
            "runtime": {"device": "cpu", "optional_dependencies": ["torch", "stable_worldmodel", "gymnasium"]},
        },
        source_path="builtin:stablewm_lewm",
    )


def runtime_status() -> AlgorithmStatus:
    import importlib.util

    details = {
        "torch_available": importlib.util.find_spec("torch") is not None,
        "stable_worldmodel_available": importlib.util.find_spec("stable_worldmodel") is not None,
        "gymnasium_available": importlib.util.find_spec("gymnasium") is not None,
    }
    available = bool(details["torch_available"] and details["stable_worldmodel_available"])
    return AlgorithmStatus(
        name="stablewm_lewm",
        available=available,
        message="available" if available else "torch and stable_worldmodel are required for stablewm_lewm.",
        details=details,
    )


class StableWMLeWMAlgorithm(AlgorithmAdapter):
    """Action scorer backed by an existing AutoCostModel-compatible checkpoint."""

    def __init__(self, manifest: AlgorithmManifest | None = None, *, cache_dir: str | Path | None = None) -> None:
        super().__init__(manifest or builtin_manifest())
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_path: Path | str | None = None
        self._checkpoint_ref: LeWMCheckpointReference | None = None
        self._cost_model: Any | None = None
        self._torch: Any | None = None

    def load(self, model_path: str | Path) -> None:
        self.model_path = model_path
        self._checkpoint_ref = normalize_lewm_checkpoint_reference(model_path, cache_dir=self.cache_dir)
        self._cost_model = None
        self._torch = None

    def score_actions(self, request: ScoreActionsRequest) -> ScoreActionsResult:
        model, torch, checkpoint_ref = self._load_cost_model()
        candidates = _action_candidates_to_tensor(request.action_candidates, torch)
        state = request.observation.vehicle_state
        info = {
            "state": torch.tensor([[[state.x, state.y, state.yaw, state.speed]]], dtype=torch.float32),
            "goal_state": torch.tensor([[[request.observation.goal[0], request.observation.goal[1]]]], dtype=torch.float32),
        }
        with torch.no_grad():
            costs = model.get_cost(info, candidates).detach().cpu().numpy().reshape(-1)
        return ScoreActionsResult(
            costs=[float(value) for value in costs],
            metadata={
                "model_path": str(self.model_path),
                "run_name": checkpoint_ref.run_name,
                "cache_dir": checkpoint_ref.cache_dir,
                "source_kind": checkpoint_ref.source_kind,
                "object_checkpoint": str(checkpoint_ref.object_checkpoint) if checkpoint_ref.object_checkpoint else None,
            },
        )

    def _load_cost_model(self) -> tuple[Any, Any, LeWMCheckpointReference]:
        if self._checkpoint_ref is None:
            raise ValueError("stablewm_lewm requires load(model_path) before score_actions().")
        if self._cost_model is not None and self._torch is not None:
            return self._cost_model, self._torch, self._checkpoint_ref
        prepare_stable_worldmodel_runtime()
        import torch
        from stable_worldmodel.policy import AutoCostModel

        self._cost_model = AutoCostModel(self._checkpoint_ref.run_name, cache_dir=self._checkpoint_ref.cache_dir)
        self._torch = torch
        return self._cost_model, self._torch, self._checkpoint_ref


def _action_candidates_to_tensor(action_candidates: Any, torch: Any) -> Any:
    rows: list[list[list[float]]] = []
    for candidate in action_candidates:
        rows.append([[float(action.steer), float(action.throttle), float(action.brake)] for action in candidate])
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("action_candidates must be a non-empty sequence of action sequences.")
    return torch.from_numpy(array[None, ...]).float()
