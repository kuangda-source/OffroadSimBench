"""Runtime registry for switchable path planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from offroad_sim.planning.types import ActionPlanner


@dataclass(slots=True)
class PlannerStatus:
    name: str
    available: bool
    message: str = ""
    details: dict[str, Any] | None = None


@dataclass(slots=True)
class PlannerSpec:
    name: str
    factory: Callable[..., ActionPlanner]
    description: str
    status_fn: Callable[[], PlannerStatus] | None = None

    def status(self) -> PlannerStatus:
        if self.status_fn is None:
            return PlannerStatus(self.name, True, "available", {})
        return self.status_fn()


class PlannerRegistry:
    """Registry that keeps planner selection out of agents and APIs."""

    def __init__(self) -> None:
        self._specs: dict[str, PlannerSpec] = {}

    def register(self, spec: PlannerSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> PlannerSpec:
        normalized = name.strip().lower().replace("-", "_")
        aliases = {"cem": "world_model_cem", "lewm": "le_wm_cem", "le_wm": "le_wm_cem"}
        normalized = aliases.get(normalized, normalized)
        try:
            return self._specs[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown planner '{name}'. Available planners: {available}") from exc

    def status(self, name: str | None = None) -> PlannerStatus | dict[str, PlannerStatus]:
        if name is not None:
            return self.get(name).status()
        return {planner_name: self._specs[planner_name].status() for planner_name in self.names()}

    def create(self, name: str, **kwargs: Any) -> ActionPlanner:
        return self.get(name).factory(**kwargs)


def default_planner_registry() -> PlannerRegistry:
    from offroad_sim.planning.cem import WorldModelCEMPlanner
    from offroad_sim.planning.stablewm import LeWMCEMPlanner

    registry = PlannerRegistry()
    registry.register(
        PlannerSpec(
            name="world_model_cem",
            factory=WorldModelCEMPlanner,
            description="Local CEM planner that evaluates candidate action sequences through BaseWorldModel.predict().",
        )
    )
    registry.register(
        PlannerSpec(
            name="le_wm_cem",
            factory=LeWMCEMPlanner,
            description="CEM planner backed by stable-worldmodel AutoCostModel checkpoints from LE-WM.",
            status_fn=_le_wm_cem_status,
        )
    )
    return registry


def _le_wm_cem_status() -> PlannerStatus:
    from offroad_sim.planning.stablewm import LeWMCEMPlanner

    details = LeWMCEMPlanner.runtime_status()
    available = bool(details["stable_worldmodel_available"] and details["torch_available"] and details["gymnasium_available"])
    return PlannerStatus(
        name="le_wm_cem",
        available=available,
        message="available" if available else "Missing stable-worldmodel planning dependencies.",
        details=details,
    )


def make_planner(name: str, **kwargs: Any) -> ActionPlanner:
    return default_planner_registry().create(name, **kwargs)
