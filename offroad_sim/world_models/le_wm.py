"""Optional LE-WM integration boundary."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Sequence

from offroad_sim.core import Action, Observation
from offroad_sim.planning.lewm_checkpoint import LeWMCheckpointFormatError, normalize_lewm_checkpoint_reference
from offroad_sim.world_models.base import BaseWorldModel, WorldModelPrediction


class LeWMUnavailableError(RuntimeError):
    """Raised when LE-WM is selected but its runtime is not configured."""


class LeWMWorldModel(BaseWorldModel):
    """Thin adapter for an externally installed LE-WM checkpoint.

    The implementation deliberately keeps lucas-maes/le-wm as an optional
    external runtime. OffroadSimBench owns the stable selection and status
    interface; checkpoint-specific inference glue can be added here without
    changing agents, CLIs, or GUI code.
    """

    model_type = "le_wm"

    def __init__(self, checkpoint_path: str | Path | None = None, *, source_dir: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        env_home = os.environ.get("LE_WM_HOME")
        self.source_dir = Path(source_dir or env_home) if source_dir or env_home else None

    @classmethod
    def runtime_status(cls, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        stable_worldmodel = importlib.util.find_spec("stable_worldmodel") is not None
        torch_available = importlib.util.find_spec("torch") is not None
        source_dir = os.environ.get("LE_WM_HOME")
        checkpoint_ref = None
        checkpoint_message = ""
        if checkpoint_path:
            try:
                checkpoint_ref = normalize_lewm_checkpoint_reference(checkpoint_path)
            except LeWMCheckpointFormatError as exc:
                checkpoint_message = str(exc)
        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        checkpoint_exists = bool(
            checkpoint_ref is not None
            and (
                checkpoint_ref.object_checkpoint is not None
                or checkpoint is None
                or not checkpoint.exists()
            )
        )
        available = stable_worldmodel and torch_available and (checkpoint_path is None or bool(checkpoint_ref))
        missing: list[str] = []
        if not stable_worldmodel:
            missing.append("stable_worldmodel package")
        if not torch_available:
            missing.append("torch package")
        if checkpoint_path is not None and checkpoint_ref is None:
            missing.append(checkpoint_message or "LE-WM checkpoint")
        return {
            "name": cls.model_type,
            "available": available,
            "message": "LE-WM runtime is ready." if available else "Missing " + ", ".join(missing) + ".",
            "details": {
                "stable_worldmodel_available": stable_worldmodel,
                "torch_available": torch_available,
                "le_wm_home": source_dir,
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "checkpoint_exists": checkpoint_exists if checkpoint else None,
                "checkpoint_run_name": checkpoint_ref.run_name if checkpoint_ref else None,
                "checkpoint_object_path": str(checkpoint_ref.object_checkpoint) if checkpoint_ref and checkpoint_ref.object_checkpoint else None,
                "checkpoint_source_kind": checkpoint_ref.source_kind if checkpoint_ref else None,
                "repository": "https://github.com/lucas-maes/le-wm",
            },
        }

    @classmethod
    def load(cls, path: str | Path) -> "LeWMWorldModel":
        model = cls(checkpoint_path=path)
        status = cls.runtime_status(path)
        if not status["available"]:
            raise LeWMUnavailableError(status["message"])
        return model

    def predict(
        self,
        observation: Observation,
        action: Action | Sequence[Action],
        horizon: int = 10,
    ) -> WorldModelPrediction:
        status = self.runtime_status(self.checkpoint_path)
        if not status["available"]:
            raise LeWMUnavailableError(
                "LE-WM is configured as an optional external model. "
                f"{status['message']} Install lucas-maes/le-wm dependencies and provide a checkpoint."
            )
        raise NotImplementedError(
            "LE-WM runtime is detected, but checkpoint-specific inference glue is intentionally "
            "kept behind this adapter. Implement Observation -> LE-WM input conversion here."
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "source_dir": str(self.source_dir) if self.source_dir else None,
            "repository": "https://github.com/lucas-maes/le-wm",
        }
