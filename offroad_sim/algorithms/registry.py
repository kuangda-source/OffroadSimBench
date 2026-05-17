"""Runtime registry for pluggable algorithm adapters."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from offroad_sim.algorithms.base import AlgorithmAdapter
from offroad_sim.algorithms.manifest import AlgorithmManifest


ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class AlgorithmStatus:
    name: str
    available: bool
    message: str = ""
    details: dict[str, Any] | None = None


@dataclass(slots=True)
class AlgorithmSpec:
    manifest: AlgorithmManifest
    factory: Callable[..., AlgorithmAdapter]
    description: str = ""
    status_fn: Callable[[], AlgorithmStatus] | None = None

    @property
    def name(self) -> str:
        return self.manifest.algorithm_id

    def status(self) -> AlgorithmStatus:
        if self.status_fn is None:
            return AlgorithmStatus(self.name, True, "available", {"source_path": self.manifest.source_path})
        return self.status_fn()


class AlgorithmRegistry:
    """Registry for built-in and local algorithm packages."""

    def __init__(self) -> None:
        self._specs: dict[str, AlgorithmSpec] = {}

    def register(self, spec: AlgorithmSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> AlgorithmSpec:
        normalized = name.strip().lower().replace("-", "_")
        try:
            return self._specs[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown algorithm '{name}'. Available algorithms: {available}") from exc

    def status(self, name: str | None = None) -> AlgorithmStatus | dict[str, AlgorithmStatus]:
        if name is not None:
            return self.get(name).status()
        return {algorithm_name: self._specs[algorithm_name].status() for algorithm_name in self.names()}

    def create(self, name: str, **kwargs: Any) -> AlgorithmAdapter:
        spec = self.get(name)
        return spec.factory(**kwargs)

    def discover_path(self, path: str | Path) -> None:
        root = Path(path)
        if not root.exists():
            return
        for manifest_path in sorted(root.glob("*/algorithm.yaml")):
            self.register(_load_package(manifest_path))


def default_algorithm_registry(search_paths: list[str | Path] | None = None) -> AlgorithmRegistry:
    from offroad_sim.algorithms.builtins.local_lewm_cost import LocalLeWMCostAlgorithm, builtin_manifest, runtime_status

    registry = AlgorithmRegistry()
    manifest = builtin_manifest()
    registry.register(
        AlgorithmSpec(
            manifest=manifest,
            factory=lambda **kwargs: LocalLeWMCostAlgorithm(manifest=manifest, **kwargs),
            description="Local LE-WM-compatible cost-model adapter.",
            status_fn=runtime_status,
        )
    )
    for search_path in search_paths if search_paths is not None else [ROOT / "algorithms"]:
        registry.discover_path(search_path)
    return registry


def make_algorithm(name: str, **kwargs: Any) -> AlgorithmAdapter:
    return default_algorithm_registry().create(name, **kwargs)


def _load_package(manifest_path: Path) -> AlgorithmSpec:
    manifest = AlgorithmManifest.from_yaml(manifest_path)
    package_dir = manifest_path.parent
    module_name, class_name = _split_entrypoint(manifest.entrypoint)
    module_path = package_dir / f"{module_name}.py"
    if not module_path.exists():
        raise ValueError(f"Algorithm entrypoint module not found: {module_path}")

    def factory(**kwargs: Any) -> AlgorithmAdapter:
        cls = _load_class(module_path, manifest.algorithm_id, class_name)
        return cls(manifest=manifest, **kwargs)

    return AlgorithmSpec(manifest=manifest, factory=factory, description=manifest.display_name)


def _split_entrypoint(entrypoint: str) -> tuple[str, str]:
    if ":" not in entrypoint:
        raise ValueError(f"Algorithm entrypoint must use module:Class format, got {entrypoint!r}")
    module_name, class_name = entrypoint.split(":", 1)
    if not module_name or not class_name:
        raise ValueError(f"Algorithm entrypoint must use module:Class format, got {entrypoint!r}")
    return module_name, class_name


def _load_class(module_path: Path, algorithm_id: str, class_name: str) -> type[AlgorithmAdapter]:
    module_key = f"offroad_sim_user_algorithm_{algorithm_id}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load algorithm module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    if not issubclass(cls, AlgorithmAdapter):
        raise TypeError(f"Algorithm class {class_name} must inherit AlgorithmAdapter.")
    return cls
