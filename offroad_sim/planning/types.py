"""Planner interfaces shared by world-model agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.world_models import BaseWorldModel


@dataclass(slots=True)
class PlanningResult:
    """Result of one receding-horizon planning query."""

    actions: list[Action]
    predicted_states: list[VehicleState] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    best_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def first_action(self) -> Action:
        return self.actions[0] if self.actions else Action()


class ActionPlanner(ABC):
    """Planner that turns an observation and model into an action sequence."""

    planner_type = "base"

    @abstractmethod
    def plan(
        self,
        observation: Observation,
        world_model: BaseWorldModel,
        *,
        reference_action: Action | None = None,
    ) -> PlanningResult:
        """Plan a short action sequence."""
