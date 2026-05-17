"""Capability-based algorithm adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from offroad_sim.core import Action, Observation, VehicleState


@dataclass(slots=True)
class AlgorithmCapabilities:
    train: bool = False
    infer: bool = False
    act: bool = False
    predict: bool = False
    score_actions: bool = False
    plan_trajectory: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "AlgorithmCapabilities":
        value = value or {}
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: bool(raw) for key, raw in value.items() if key in allowed})

    def available_names(self) -> list[str]:
        return [key for key in self.__dataclass_fields__ if bool(getattr(self, key))]


@dataclass(slots=True)
class DataPrepRequest:
    episode_root: str | Path
    output_path: str | Path
    actions_from_state: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataPrepResult:
    output_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"output_path": self.output_path, **self.metadata}


@dataclass(slots=True)
class TrainRequest:
    input_path: str | Path
    output_dir: str | Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainResult:
    output_dir: str
    checkpoint_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"output_dir": self.output_dir, "checkpoint_path": self.checkpoint_path, **self.metadata}


@dataclass(slots=True)
class ActRequest:
    observation: Observation
    task_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictRequest:
    observation: Observation
    actions: Action | Sequence[Action]
    horizon: int = 10
    task_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreActionsRequest:
    observation: Observation
    action_candidates: Sequence[Sequence[Action]]
    task_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreActionsResult:
    costs: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryPlanRequest:
    observation: Observation
    task_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryPlanResult:
    states: list[VehicleState]
    metadata: dict[str, Any] = field(default_factory=dict)


class UnsupportedCapabilityError(NotImplementedError):
    """Raised when an adapter method is called but not declared/supported."""


class AlgorithmAdapter:
    """Base class for optional-capability algorithm packages."""

    def __init__(self, manifest: Any) -> None:
        self.manifest = manifest
        self.algorithm_id = str(manifest.algorithm_id)
        self.capabilities: AlgorithmCapabilities = manifest.capabilities

    def prepare_data(self, request: DataPrepRequest) -> DataPrepResult:
        self._raise_unsupported("prepare_data")

    def train(self, request: TrainRequest) -> TrainResult:
        self._raise_unsupported("train")

    def load(self, model_path: str | Path) -> None:
        self._raise_unsupported("load")

    def act(self, request: ActRequest) -> Action:
        self._raise_unsupported("act")

    def predict(self, request: PredictRequest) -> Any:
        self._raise_unsupported("predict")

    def score_actions(self, request: ScoreActionsRequest) -> ScoreActionsResult:
        self._raise_unsupported("score_actions")

    def plan_trajectory(self, request: TrajectoryPlanRequest) -> TrajectoryPlanResult:
        self._raise_unsupported("plan_trajectory")

    def _raise_unsupported(self, capability: str) -> Any:
        available = ", ".join(self.capabilities.available_names()) or "none"
        raise UnsupportedCapabilityError(
            f"Algorithm '{self.algorithm_id}' does not support {capability}. Available capabilities: {available}."
        )
