"""Built-in adapter for the local LE-WM-compatible cost model workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.algorithms.base import (
    AlgorithmAdapter,
    DataPrepRequest,
    DataPrepResult,
    ScoreActionsRequest,
    ScoreActionsResult,
    TrainRequest,
    TrainResult,
)
from offroad_sim.algorithms.manifest import AlgorithmManifest
from offroad_sim.algorithms.registry import AlgorithmStatus
from offroad_sim.utils.runtime_env import prepare_stable_worldmodel_runtime


ROOT = Path(__file__).resolve().parents[3]


def builtin_manifest() -> AlgorithmManifest:
    return AlgorithmManifest.from_dict(
        {
            "algorithm_id": "local_lewm_cost",
            "display_name": "Local LE-WM Cost Model",
            "entrypoint": "offroad_sim.algorithms.builtins.local_lewm_cost:LocalLeWMCostAlgorithm",
            "version": "0.1.0",
            "capabilities": {
                "train": True,
                "infer": True,
                "score_actions": True,
            },
            "input_contract": {
                "observations": ["state", "goal", "rgb_optional"],
                "actions": ["steer", "throttle", "brake"],
                "task": "navigation_region_v1",
            },
            "output_contract": {"mode": "action_cost"},
            "runtime": {"device": "cpu", "optional_dependencies": ["torch", "stable_worldmodel"]},
        },
        source_path="builtin:local_lewm_cost",
    )


def runtime_status() -> AlgorithmStatus:
    import importlib.util

    details = {
        "torch_available": importlib.util.find_spec("torch") is not None,
        "stable_worldmodel_available": importlib.util.find_spec("stable_worldmodel") is not None,
    }
    return AlgorithmStatus(
        name="local_lewm_cost",
        available=details["torch_available"],
        message="available" if details["torch_available"] else "torch is required for local_lewm_cost training/inference.",
        details=details,
    )


class LocalLeWMCostAlgorithm(AlgorithmAdapter):
    """Adapter that exposes the existing StableWM HDF5 and cost checkpoint flow."""

    def __init__(self, manifest: AlgorithmManifest | None = None, command_runner: Any | None = None) -> None:
        super().__init__(manifest or builtin_manifest())
        self.command_runner = command_runner or _run_json_command
        self.model_path: Path | None = None
        self._cost_model: Any | None = None
        self._torch: Any | None = None

    def prepare_data(self, request: DataPrepRequest) -> DataPrepResult:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "export_episodes_hdf5.py"),
            str(request.episode_root),
            str(request.output_path),
        ]
        if request.actions_from_state:
            command.append("--actions-from-state")
        payload = self.command_runner(command)
        return DataPrepResult(
            output_path=str(Path(payload.get("output_hdf5", request.output_path)).resolve()),
            metadata={key: value for key, value in payload.items() if key != "output_hdf5"},
        )

    def train(self, request: TrainRequest) -> TrainResult:
        payload = self.command_runner(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_lewm_cost_model.py"),
                str(request.input_path),
                "--output",
                str(request.output_dir),
            ]
        )
        return TrainResult(
            output_dir=str(Path(payload.get("output_dir", request.output_dir)).resolve()),
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            metadata={key: value for key, value in payload.items() if key not in {"output_dir", "checkpoint_path"}},
        )

    def load(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._cost_model = None
        self._torch = None

    def score_actions(self, request: ScoreActionsRequest) -> ScoreActionsResult:
        model, torch = self._load_cost_model()
        candidates = _action_candidates_to_tensor(request.action_candidates, torch)
        state = request.observation.vehicle_state
        info = {
            "state": torch.tensor([[[state.x, state.y, state.yaw, state.speed]]], dtype=torch.float32),
            "goal_state": torch.tensor([[[request.observation.goal[0], request.observation.goal[1]]]], dtype=torch.float32),
        }
        with torch.no_grad():
            costs = model.get_cost(info, candidates).detach().cpu().numpy().reshape(-1)
        return ScoreActionsResult(costs=[float(value) for value in costs], metadata={"model_path": str(self.model_path)})

    def _load_cost_model(self) -> tuple[Any, Any]:
        if self.model_path is None:
            raise ValueError("local_lewm_cost requires load(model_path) before score_actions().")
        if self._cost_model is not None and self._torch is not None:
            return self._cost_model, self._torch
        prepare_stable_worldmodel_runtime()
        import torch

        checkpoint = self.model_path
        if checkpoint.is_dir():
            checkpoint = checkpoint / "lewm_cost_object.ckpt"
        self._cost_model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
        self._torch = torch
        return self._cost_model, self._torch


def _run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "command failed").strip())
    output = completed.stdout.strip()
    json_start = output.find("{")
    if json_start < 0:
        raise RuntimeError(f"Command did not emit JSON: {' '.join(command)}")
    payload, _ = json.JSONDecoder().raw_decode(output[json_start:])
    return payload


def _action_candidates_to_tensor(action_candidates: Any, torch: Any) -> Any:
    rows: list[list[list[float]]] = []
    for candidate in action_candidates:
        rows.append([[float(action.steer), float(action.throttle), float(action.brake)] for action in candidate])
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("action_candidates must be a non-empty sequence of action sequences.")
    return torch.from_numpy(array[None, ...]).float()
