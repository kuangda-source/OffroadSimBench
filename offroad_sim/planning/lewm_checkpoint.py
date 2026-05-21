"""Helpers for stable-worldmodel / upstream LE-WM checkpoint references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class LeWMCheckpointFormatError(ValueError):
    """Raised when a provided LE-WM checkpoint cannot be loaded directly."""


@dataclass(frozen=True, slots=True)
class LeWMCheckpointReference:
    """Normalized checkpoint reference accepted by stable_worldmodel.AutoCostModel."""

    run_name: str
    cache_dir: str | None = None
    object_checkpoint: Path | None = None
    source_kind: str = "stablewm_run_name"


def normalize_lewm_checkpoint_reference(
    checkpoint: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> LeWMCheckpointReference:
    """Normalize user-provided LE-WM checkpoint paths for AutoCostModel.

    stable-worldmodel accepts either a run directory containing `*_object.ckpt`
    or a run name/stem without the `_object.ckpt` suffix. Users often paste the
    direct checkpoint file path, so this helper strips that suffix.
    """

    raw = Path(checkpoint)
    normalized_cache = str(Path(cache_dir)) if cache_dir is not None else None
    if raw.exists():
        raw = raw.resolve()
        if raw.is_dir():
            object_files = sorted(raw.glob("*_object.ckpt"), key=lambda path: path.stat().st_ctime, reverse=True)
            if object_files:
                return LeWMCheckpointReference(
                    run_name=str(raw),
                    cache_dir=normalized_cache,
                    object_checkpoint=object_files[0],
                    source_kind="stablewm_run_dir",
                )
            if (raw / "weights.pt").exists() and (raw / "config.json").exists():
                raise LeWMCheckpointFormatError(
                    "This looks like a HuggingFace LE-WM weights directory. Convert it first with "
                    "scripts/convert_lewm_hf_checkpoint.py so stable_worldmodel can load an *_object.ckpt file."
                )
            raise LeWMCheckpointFormatError(f"No *_object.ckpt file found in LE-WM run directory: {raw}")
        if raw.name.endswith("_object.ckpt"):
            run_stem = raw.with_name(raw.name[: -len("_object.ckpt")])
            return LeWMCheckpointReference(
                run_name=str(run_stem),
                cache_dir=normalized_cache,
                object_checkpoint=raw,
                source_kind="stablewm_object_file",
            )
        raise LeWMCheckpointFormatError(
            f"Unsupported LE-WM checkpoint file {raw}. Expected a stable-worldmodel *_object.ckpt file."
        )

    text = str(checkpoint)
    if text.endswith("_object.ckpt"):
        text = text[: -len("_object.ckpt")]
    return LeWMCheckpointReference(run_name=text, cache_dir=normalized_cache)
