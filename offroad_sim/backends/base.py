"""Base interface for simulator backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from offroad_sim.core.types import Action, Observation, StepResult


class OffroadSimBackend(ABC):
    """Common simulator backend interface used by all agents."""

    @abstractmethod
    def reset(self, scenario_config: Any) -> Observation:
        """Reset the backend to the beginning of a scenario."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Advance the backend by one control step."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Return the latest observation without stepping."""

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return backend-level metrics collected so far."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

