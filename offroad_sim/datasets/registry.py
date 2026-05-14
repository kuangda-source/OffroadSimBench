"""Registry used to resolve dataset adapters at runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from offroad_sim.datasets.adapters import DatasetAdapter, ORFDAdapter, OffroadSimV1Adapter


class DatasetRegistry:
    """Small adapter registry with explicit lookup and auto-detection."""

    def __init__(self, adapters: Iterable[DatasetAdapter] | None = None) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        for adapter in adapters or ():
            self.register(adapter)

    def register(self, adapter: DatasetAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def get(self, name: str) -> DatasetAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown dataset adapter '{name}'. Available adapters: {available}") from exc

    def resolve(self, dataset_root: str | Path, adapter_name: str | None = None) -> DatasetAdapter:
        if adapter_name:
            return self.get(adapter_name)

        root = Path(dataset_root)
        for adapter in sorted(self._adapters.values(), key=lambda item: item.priority):
            if adapter.can_load(root):
                return adapter

        available = ", ".join(self.names()) or "none"
        raise ValueError(f"No registered dataset adapter can load {root}. Available adapters: {available}")


def default_dataset_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register(OffroadSimV1Adapter())
    registry.register(ORFDAdapter())
    return registry
