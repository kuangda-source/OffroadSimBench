"""Base interface for all driving agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from offroad_sim.core.types import Action, Observation


class OffroadAgent(ABC):
    """Simulator-neutral driving policy interface."""

    def reset(self, scenario_info: Any) -> None:
        """Reset agent state before a new episode starts."""

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """Return the next driving action for the current observation."""

    def close(self) -> None:
        """Release agent resources."""

