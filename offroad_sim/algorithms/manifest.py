"""Algorithm manifest parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from offroad_sim.algorithms.base import AlgorithmCapabilities
from offroad_sim.utils.yaml_io import load_yaml_file


@dataclass(slots=True)
class AlgorithmManifest:
    algorithm_id: str
    display_name: str
    entrypoint: str
    version: str = "0.1.0"
    capabilities: AlgorithmCapabilities = field(default_factory=AlgorithmCapabilities)
    input_contract: dict[str, Any] = field(default_factory=dict)
    output_contract: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str | Path = "") -> "AlgorithmManifest":
        for key in ("algorithm_id", "entrypoint"):
            if not str(data.get(key, "")).strip():
                raise ValueError(f"Algorithm manifest missing required field: {key}")
        algorithm_id = _normalize_id(str(data["algorithm_id"]))
        display_name = str(data.get("display_name") or algorithm_id)
        return cls(
            algorithm_id=algorithm_id,
            display_name=display_name,
            entrypoint=str(data["entrypoint"]),
            version=str(data.get("version", "0.1.0")),
            capabilities=AlgorithmCapabilities.from_mapping(data.get("capabilities")),
            input_contract=dict(data.get("input_contract", {})),
            output_contract=dict(data.get("output_contract", {})),
            runtime=dict(data.get("runtime", {})),
            source_path=str(source_path),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlgorithmManifest":
        manifest_path = Path(path)
        return cls.from_dict(load_yaml_file(manifest_path), source_path=manifest_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "display_name": self.display_name,
            "entrypoint": self.entrypoint,
            "version": self.version,
            "capabilities": {key: bool(getattr(self.capabilities, key)) for key in self.capabilities.__dataclass_fields__},
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "runtime": self.runtime,
            "source_path": self.source_path,
        }


def _normalize_id(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Algorithm id cannot be empty.")
    return normalized
