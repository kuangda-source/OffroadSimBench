"""Runtime registry for switchable driving algorithms."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from offroad_sim.agents.base import OffroadAgent


@dataclass(slots=True)
class AgentSpec:
    name: str
    factory: Callable[..., OffroadAgent]
    description: str


class AgentRegistry:
    """Registry that keeps algorithm selection out of application code."""

    def __init__(self) -> None:
        self._specs: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> AgentSpec:
        normalized = name.strip().lower().replace("-", "_")
        aliases = {"goal": "rule_based", "rulebased": "rule_based", "worldmodel": "world_model", "wm": "world_model"}
        normalized = aliases.get(normalized, normalized)
        try:
            return self._specs[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise ValueError(f"Unknown agent '{name}'. Available agents: {available}") from exc

    def create(self, name: str, **kwargs: Any) -> OffroadAgent:
        spec = self.get(name)
        signature = inspect.signature(spec.factory)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return spec.factory(**kwargs)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return spec.factory(**accepted)


def default_agent_registry() -> AgentRegistry:
    from offroad_sim.agents.basic import KeyboardAgent, RandomAgent, RuleBasedGoalAgent, StopAgent
    from offroad_sim.agents.world_model import WorldModelAgent

    registry = AgentRegistry()
    registry.register(AgentSpec("keyboard", KeyboardAgent, "Placeholder for interactive keyboard driving."))
    registry.register(AgentSpec("random", RandomAgent, "Random action baseline."))
    registry.register(AgentSpec("rule_based", RuleBasedGoalAgent, "Goal follower with terrain-risk slowdown."))
    registry.register(AgentSpec("stop", StopAgent, "Always command a full stop."))
    registry.register(AgentSpec("world_model", WorldModelAgent, "Rule-based controller with switchable world-model risk checks."))
    return registry


def make_agent(name: str, **kwargs: Any) -> OffroadAgent:
    return default_agent_registry().create(name, **kwargs)
