"""Small YAML loading helper for configuration modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML document and require a mapping at the top level."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load OffroadSimBench YAML configs. "
            "Install project dependencies with `python -m pip install -e .`."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")

    return data

