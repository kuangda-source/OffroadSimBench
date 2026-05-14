"""World model interfaces shared by planners and learned models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from offroad_sim.core.types import Action, Observation, VehicleState


@dataclass(slots=True)
class WorldModelPrediction:
    """Predicted future states and auxiliary model outputs."""

    states: list[VehicleState]
    actions: list[Action] = field(default_factory=list)
    risk_map: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_state(self) -> VehicleState | None:
        return self.states[-1] if self.states else None


class BaseWorldModel(ABC):
    """Abstract interface for analytic and learned world models."""

    model_type = "base"

    def reset(self, scenario_info: dict[str, Any] | None = None) -> None:
        """Reset any recurrent state before a new episode."""

        return None

    @abstractmethod
    def predict(
        self,
        observation: Observation,
        action: Action | Sequence[Action],
        horizon: int = 10,
    ) -> WorldModelPrediction:
        """Predict a short rollout from the current observation."""

    def get_config(self) -> dict[str, Any]:
        return {}

    def save(self, path: str | Path) -> Path:
        """Save lightweight model metadata to disk."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "config": self.get_config(),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "BaseWorldModel":
        """Load a model from disk when the concrete class supports it."""

        raise NotImplementedError(f"{cls.__name__} does not implement load().")


def vehicle_state_to_dict(state: VehicleState) -> dict[str, Any]:
    return asdict(state)
