"""Runtime registry for switchable world models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from offroad_sim.world_models.base import BaseWorldModel


@dataclass(slots=True)
class WorldModelStatus:
    name: str
    available: bool
    message: str = ""
    details: dict[str, Any] | None = None


@dataclass(slots=True)
class WorldModelSpec:
    name: str
    factory: Callable[..., BaseWorldModel]
    description: str
    load_fn: Callable[[str | Path], BaseWorldModel] | None = None
    status_fn: Callable[[], WorldModelStatus] | None = None

    def status(self) -> WorldModelStatus:
        if self.status_fn is None:
            return WorldModelStatus(self.name, True, "available", {})
        return self.status_fn()


class WorldModelRegistry:
    """Registry that keeps model selection out of callers and agents."""

    def __init__(self) -> None:
        self._specs: dict[str, WorldModelSpec] = {}

    def register(self, spec: WorldModelSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> WorldModelSpec:
        normalized = name.strip().lower().replace("-", "_")
        aliases = {"wm": "tiny_learned", "tiny": "tiny_learned", "kinematic": "simple_kinematic", "le_wm": "le_wm"}
        normalized = aliases.get(normalized, normalized)
        try:
            return self._specs[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown world model '{name}'. Available world models: {available}") from exc

    def status(self, name: str | None = None) -> WorldModelStatus | dict[str, WorldModelStatus]:
        if name is not None:
            return self.get(name).status()
        return {model_name: self._specs[model_name].status() for model_name in self.names()}

    def create(self, name: str, *, path: str | Path | None = None, **kwargs: Any) -> BaseWorldModel:
        spec = self.get(name)
        if path is not None:
            if spec.load_fn is None:
                raise ValueError(f"World model '{spec.name}' does not support loading from a path.")
            return spec.load_fn(path)
        return spec.factory(**kwargs)


def default_world_model_registry() -> WorldModelRegistry:
    from offroad_sim.world_models.kinematic import SimpleKinematicWorldModel
    from offroad_sim.world_models.le_wm import LeWMWorldModel
    from offroad_sim.world_models.tiny_learned import TinyLearnedWorldModel

    registry = WorldModelRegistry()
    registry.register(
        WorldModelSpec(
            name="simple_kinematic",
            factory=SimpleKinematicWorldModel,
            load_fn=SimpleKinematicWorldModel.load,
            description="Analytic bicycle-model rollout used as a deterministic baseline.",
        )
    )
    registry.register(
        WorldModelSpec(
            name="tiny_learned",
            factory=TinyLearnedWorldModel,
            load_fn=TinyLearnedWorldModel.load,
            description="Small NumPy learned dynamics model trained from dataset sequences.",
        )
    )
    registry.register(
        WorldModelSpec(
            name="le_wm",
            factory=LeWMWorldModel,
            load_fn=LeWMWorldModel.load,
            description="Optional wrapper for lucas-maes/le-wm checkpoints and runtime.",
            status_fn=lambda: WorldModelStatus(**LeWMWorldModel.runtime_status()),
        )
    )
    return registry


def make_world_model(name: str, *, path: str | Path | None = None, **kwargs: Any) -> BaseWorldModel:
    return default_world_model_registry().create(name, path=path, **kwargs)
