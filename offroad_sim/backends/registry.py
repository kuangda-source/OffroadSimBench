"""Runtime registry for simulator backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from offroad_sim.backends.base import OffroadSimBackend


@dataclass(slots=True)
class BackendStatus:
    """Human-readable availability state for one backend."""

    name: str
    available: bool
    message: str = ""
    details: dict[str, Any] | None = None


@dataclass(slots=True)
class BackendSpec:
    """Factory metadata for one backend implementation."""

    name: str
    factory: Callable[..., OffroadSimBackend]
    description: str
    status_fn: Callable[[], BackendStatus] | None = None

    def status(self) -> BackendStatus:
        if self.status_fn is None:
            return BackendStatus(self.name, True, "available", {})
        return self.status_fn()


class BackendRegistry:
    """Registry that keeps backend creation out of application code."""

    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}

    def register(self, spec: BackendSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> BackendSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown backend '{name}'. Available backends: {available}") from exc

    def status(self, name: str | None = None) -> BackendStatus | dict[str, BackendStatus]:
        if name is not None:
            return self.get(name).status()
        return {backend_name: self._specs[backend_name].status() for backend_name in self.names()}

    def create(self, name: str, **kwargs: Any) -> OffroadSimBackend:
        return self.get(name).factory(**kwargs)


def default_backend_registry() -> BackendRegistry:
    from offroad_sim.backends.beamng_backend import BeamNGBackend
    from offroad_sim.backends.dataset_replay_backend import DatasetReplayBackend
    from offroad_sim.backends.gym_heightmap_backend import GymHeightmapBackend
    from offroad_sim.backends.ue5_backend import UE5Backend

    registry = BackendRegistry()
    registry.register(
        BackendSpec(
            name="gym_heightmap",
            factory=GymHeightmapBackend,
            description="Lightweight procedural 2.5D training and testing backend.",
        )
    )
    registry.register(
        BackendSpec(
            name="dataset_replay",
            factory=DatasetReplayBackend,
            description="Dataset sequence replay backend with adapter-based loading.",
        )
    )
    registry.register(
        BackendSpec(
            name="beamng",
            factory=BeamNGBackend,
            description="Optional BeamNG.tech backend using beamngpy.",
            status_fn=BeamNGBackend.runtime_status,
        )
    )
    registry.register(
        BackendSpec(
            name="ue5",
            factory=UE5Backend,
            description="TCP JSON bridge for a future UE5 runtime or the local mock server.",
        )
    )
    return registry


def make_backend(name: str, **kwargs: Any) -> OffroadSimBackend:
    """Create a backend from the default registry."""

    return default_backend_registry().create(name, **kwargs)
