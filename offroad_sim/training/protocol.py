"""Versioned, simulator-neutral trainer manifest protocol."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


TRAINER_SCHEMA_VERSION = 1
LAUNCH_KINDS = {"python_script", "python_module", "executable"}
PARAMETER_TYPES = {
    "str",
    "string",
    "path",
    "file",
    "directory",
    "int",
    "integer",
    "float",
    "number",
    "bool",
    "boolean",
}


@dataclass(frozen=True, slots=True)
class TrainerLaunch:
    """Normalized process launch description."""

    kind: str
    entrypoint: str = ""
    module: str = ""
    conda_env: str = ""
    working_directory: str = ""
    environment: dict[str, str] | None = None


def normalize_trainer_manifest(data: Mapping[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    """Normalize schema-v1 and legacy trainer manifests into one contract."""

    schema_version = int(data.get("schema_version") or TRAINER_SCHEMA_VERSION)
    if schema_version != TRAINER_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trainer schema_version {schema_version}; expected {TRAINER_SCHEMA_VERSION}."
        )

    trainer_id = str(data.get("trainer_id") or data.get("id") or manifest_path.parent.name).strip()
    if not trainer_id:
        raise ValueError(f"Trainer manifest has no trainer_id: {manifest_path}")

    raw_launch = data.get("launch") if isinstance(data.get("launch"), Mapping) else {}
    legacy_runtime = str(data.get("runtime") or "python").strip().lower()
    default_kind = "python_script" if legacy_runtime == "python" else "executable"
    kind = str(raw_launch.get("kind") or default_kind).strip().lower()
    if kind not in LAUNCH_KINDS:
        raise ValueError(f"Unsupported trainer launch kind: {kind}")

    entrypoint = str(raw_launch.get("entrypoint") or data.get("entrypoint") or "").strip()
    module = str(raw_launch.get("module") or data.get("module") or "").strip()
    if kind == "python_module" and not module:
        raise ValueError(f"Python-module trainer has no module: {manifest_path}")
    if kind != "python_module" and not entrypoint:
        raise ValueError(f"Trainer manifest has no entrypoint: {manifest_path}")

    raw_environment = raw_launch.get("environment", data.get("environment", {}))
    if raw_environment is None:
        raw_environment = {}
    if not isinstance(raw_environment, Mapping):
        raise ValueError("launch.environment must be a mapping.")

    parameters = data.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping.")
    normalized_parameters = _normalize_parameter_schema(parameters)

    raw_input = data.get("input") or {}
    if not isinstance(raw_input, Mapping):
        raise ValueError("input must be a mapping.")
    input_spec = _normalize_input_spec(raw_input)
    inference = _normalize_inference_spec(data.get("inference"), manifest_path=manifest_path)

    launch = {
        "kind": kind,
        "entrypoint": entrypoint,
        "module": module,
        "conda_env": str(raw_launch.get("conda_env") or data.get("conda_env") or "").strip(),
        "working_directory": str(
            raw_launch.get("working_directory") or data.get("working_directory") or ""
        ).strip(),
        "environment": {str(key): str(value) for key, value in raw_environment.items()},
    }
    return {
        "schema_version": schema_version,
        "trainer_id": trainer_id,
        "display_name": str(data.get("display_name") or data.get("label") or trainer_id),
        "description": str(data.get("description") or ""),
        "launch": launch,
        "arguments": [str(value) for value in (data.get("arguments") or [])],
        "parameters": normalized_parameters,
        "input": input_spec,
        "outputs": dict(data.get("outputs") or {}),
        "inference": inference,
    }


def validate_trainer_parameters(schema: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce parameter values and enforce schema constraints/dependencies."""

    values: dict[str, Any] = {}
    normalized_overrides = {str(name): value for name, value in overrides.items()}
    unknown = sorted(set(normalized_overrides) - set(schema))
    if unknown:
        raise ValueError(f"Unknown trainer parameter(s): {', '.join(unknown)}")

    for raw_name, raw_spec in schema.items():
        name = str(raw_name)
        spec = dict(raw_spec) if isinstance(raw_spec, Mapping) else {}
        active = parameter_dependency_satisfied(spec, {**values, **normalized_overrides})
        if name in normalized_overrides:
            raw_value = normalized_overrides[name]
        elif "default" in spec:
            raw_value = spec.get("default")
        elif spec.get("required") is True and active:
            raise ValueError(f"Missing required parameter: {name}")
        else:
            continue
        if not active:
            continue
        value = _coerce_value(raw_value, str(spec.get("type") or "str"))
        _validate_parameter_value(name, value, spec)
        values[name] = value
    return values


def parameter_dependency_satisfied(spec: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    dependency = spec.get("depends_on")
    if dependency in (None, "", {}):
        return True
    if isinstance(dependency, str):
        return bool(values.get(dependency))
    if not isinstance(dependency, Mapping):
        raise ValueError("depends_on must be a parameter name or mapping.")
    name = str(dependency.get("parameter") or dependency.get("name") or "").strip()
    if not name:
        raise ValueError("depends_on mapping requires 'parameter'.")
    current = values.get(name)
    if "equals" in dependency:
        return current == dependency.get("equals")
    if "not_equals" in dependency:
        return current != dependency.get("not_equals")
    return bool(current)


def build_trainer_command(
    manifest: Mapping[str, Any],
    *,
    arguments: list[str],
    manifest_dir: Path,
) -> list[str]:
    """Build a process command for Python scripts/modules or executables."""

    launch = dict(manifest.get("launch") or {})
    kind = str(launch.get("kind") or "python_script")
    conda_env = str(launch.get("conda_env") or "").strip()
    prefix: list[str] = []
    python_executable = sys.executable
    using_conda_run = False
    if conda_env:
        env_python = _conda_python(conda_env, manifest_dir)
        if env_python is not None:
            python_executable = str(env_python)
        else:
            conda = shutil.which("conda")
            if not conda:
                raise FileNotFoundError(f"Conda executable not found for environment: {conda_env}")
            selector = "-p" if _looks_like_path(conda_env) else "-n"
            target = str(_resolve_path(manifest_dir, conda_env)) if selector == "-p" else conda_env
            prefix = [conda, "run", selector, target]
            python_executable = "python"
            using_conda_run = True

    if kind == "python_module":
        command = [python_executable, "-m", str(launch.get("module") or "")]
    elif kind == "python_script":
        command = [python_executable, str(_resolve_path(manifest_dir, str(launch.get("entrypoint") or "")))]
    elif kind == "executable":
        entrypoint = str(launch.get("entrypoint") or "")
        executable = entrypoint if using_conda_run and not _looks_like_path(entrypoint) else _resolve_executable(manifest_dir, entrypoint)
        command = [str(executable)]
    else:
        raise ValueError(f"Unsupported trainer launch kind: {kind}")
    return [*prefix, *command, *arguments]


def resolve_trainer_working_directory(manifest: Mapping[str, Any], manifest_dir: Path) -> Path:
    launch = dict(manifest.get("launch") or {})
    value = str(launch.get("working_directory") or "").strip()
    target = _resolve_path(manifest_dir, value) if value else manifest_dir.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Trainer working directory not found: {target}")
    return target


def resolve_trainer_environment(manifest: Mapping[str, Any], manifest_dir: Path) -> dict[str, str]:
    launch = dict(manifest.get("launch") or {})
    configured = launch.get("environment") or {}
    context = {
        "manifest_dir": str(manifest_dir.resolve()),
        "pathsep": os.pathsep,
    }
    environment = dict(os.environ)
    for key, value in dict(configured).items():
        environment[str(key)] = str(value).format_map(context)
    return environment


def _normalize_parameter_schema(parameters: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in parameters.items():
        name = str(raw_name)
        if not name:
            raise ValueError("Parameter names cannot be empty.")
        spec = dict(raw_spec) if isinstance(raw_spec, Mapping) else {"default": raw_spec}
        value_type = str(spec.get("type") or "str").lower()
        if value_type not in PARAMETER_TYPES:
            raise ValueError(f"Unsupported parameter type for {name}: {value_type}")
        spec["type"] = value_type
        if "enum" in spec and not isinstance(spec["enum"], list):
            raise ValueError(f"Parameter enum must be a list: {name}")
        if "depends_on" in spec:
            parameter_dependency_satisfied(spec, {})
        normalized[name] = spec
    for name, spec in normalized.items():
        dependency = spec.get("depends_on")
        if isinstance(dependency, str):
            dependency_name = dependency
        elif isinstance(dependency, Mapping):
            dependency_name = str(dependency.get("parameter") or dependency.get("name") or "")
        else:
            dependency_name = ""
        if dependency_name and dependency_name not in normalized:
            raise ValueError(f"Parameter {name} depends on unknown parameter: {dependency_name}")
        if "default" in spec:
            value = _coerce_value(spec["default"], str(spec["type"]))
            _validate_parameter_value(name, value, spec)
            spec["default"] = value
    return normalized


def _normalize_input_spec(raw_input: Mapping[str, Any]) -> dict[str, Any]:
    dataset_format = raw_input.get("dataset_format", "any_registered_adapter")
    if isinstance(dataset_format, list):
        formats = [str(value) for value in dataset_format]
        dataset_format = formats
    else:
        dataset_format = str(dataset_format)
    required_modalities = raw_input.get("required_modalities") or []
    optional_modalities = raw_input.get("optional_modalities") or []
    if not isinstance(required_modalities, list) or not isinstance(optional_modalities, list):
        raise ValueError("required_modalities and optional_modalities must be lists.")
    return {
        **dict(raw_input),
        "dataset_format": dataset_format,
        "required_modalities": [str(value) for value in required_modalities],
        "optional_modalities": [str(value) for value in optional_modalities],
        "split_required": bool(raw_input.get("split_required", False)),
    }


def _normalize_inference_spec(raw: Any, *, manifest_path: Path) -> dict[str, Any]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("inference must be a mapping.")
    raw_launch = raw.get("launch") if isinstance(raw.get("launch"), Mapping) else {}
    kind = str(raw_launch.get("kind") or "python_script").strip().lower()
    if kind not in LAUNCH_KINDS:
        raise ValueError(f"Unsupported inference launch kind: {kind}")
    entrypoint = str(raw_launch.get("entrypoint") or "").strip()
    module = str(raw_launch.get("module") or "").strip()
    if kind == "python_module" and not module:
        raise ValueError(f"Python-module inference has no module: {manifest_path}")
    if kind != "python_module" and not entrypoint:
        raise ValueError(f"Inference has no entrypoint: {manifest_path}")
    raw_environment = raw_launch.get("environment") or {}
    if not isinstance(raw_environment, Mapping):
        raise ValueError("inference.launch.environment must be a mapping.")
    parameters = raw.get("parameters") or {}
    input_spec = raw.get("input") or {}
    if not isinstance(parameters, Mapping) or not isinstance(input_spec, Mapping):
        raise ValueError("inference parameters and input must be mappings.")
    return {
        "launch": {
            "kind": kind,
            "entrypoint": entrypoint,
            "module": module,
            "conda_env": str(raw_launch.get("conda_env") or "").strip(),
            "working_directory": str(raw_launch.get("working_directory") or "").strip(),
            "environment": {str(key): str(value) for key, value in raw_environment.items()},
        },
        "arguments": [str(value) for value in (raw.get("arguments") or [])],
        "parameters": _normalize_parameter_schema(parameters),
        "input": _normalize_input_spec(input_spec),
        "outputs": dict(raw.get("outputs") or {}),
    }


def _coerce_value(value: Any, value_type: str) -> Any:
    normalized = value_type.lower()
    if normalized in {"int", "integer"}:
        return int(value)
    if normalized in {"float", "number"}:
        return float(value)
    if normalized in {"bool", "boolean"}:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")
    return str(value) if value is not None else ""


def _validate_parameter_value(name: str, value: Any, spec: Mapping[str, Any]) -> None:
    choices = spec.get("enum")
    if choices is not None and value not in choices:
        raise ValueError(f"Parameter {name} must be one of: {', '.join(map(str, choices))}")
    if "min" in spec and value < _coerce_value(spec["min"], str(spec.get("type") or "float")):
        raise ValueError(f"Parameter {name} must be >= {spec['min']}")
    if "max" in spec and value > _coerce_value(spec["max"], str(spec.get("type") or "float")):
        raise ValueError(f"Parameter {name} must be <= {spec['max']}")


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _resolve_executable(base_dir: Path, value: str) -> Path | str:
    if not _looks_like_path(value):
        found = shutil.which(value)
        if found:
            return Path(found).resolve()
    return _resolve_path(base_dir, value)


def _looks_like_path(value: str) -> bool:
    return any(separator in value for separator in ("/", "\\")) or value.startswith((".", "~")) or bool(Path(value).drive)


def _conda_python(conda_env: str, manifest_dir: Path) -> Path | None:
    if not _looks_like_path(conda_env):
        return None
    root = _resolve_path(manifest_dir, conda_env)
    candidates = (root / "python.exe", root / "bin" / "python")
    return next((candidate for candidate in candidates if candidate.is_file()), None)
