"""Dataset quality analysis and deterministic split utilities."""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from offroad_sim.datasets.types import DatasetFrame, DatasetSequence


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class DatasetAnalysisOptions:
    """Bound expensive validation while keeping the report reproducible."""

    max_asset_checks: int = 2000
    max_reported_issues: int = 200
    max_disk_files: int = 200000


def analyze_dataset_sequences(
    sequences: list[DatasetSequence],
    *,
    dataset_root: str | Path,
    options: DatasetAnalysisOptions | None = None,
) -> dict[str, Any]:
    """Return model-neutral dataset statistics and quality findings."""

    settings = options or DatasetAnalysisOptions()
    root = Path(dataset_root).resolve()
    modality_counts: dict[str, int] = {}
    declared_modalities: set[str] = set()
    modality_shapes: dict[str, set[tuple[int, ...]]] = {}
    missing_counts: dict[str, int] = {}
    sequence_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    unique_assets: dict[str, str] = {}
    timestamps: list[float] = []
    total_frames = 0

    for sequence in sequences:
        total_frames += len(sequence.frames)
        timestamps.extend(frame.timestamp for frame in sequence.frames if math.isfinite(float(frame.timestamp)))
        expected_modalities = _expected_modalities(sequence)
        declared_modalities.update(expected_modalities)
        per_sequence_counts = {name: 0 for name in expected_modalities}
        for frame in sequence.frames:
            assets = frame.available_assets()
            for name in expected_modalities:
                path_value = assets.get(name)
                if not path_value:
                    missing_counts[name] = missing_counts.get(name, 0) + 1
                    _append_issue(
                        issues,
                        settings,
                        severity="error",
                        code="missing_asset",
                        sequence_id=sequence.sequence_id,
                        frame_id=frame.frame_id,
                        modality=name,
                        message=f"{name} is missing for frame {frame.frame_id}.",
                    )
                    continue
                per_sequence_counts[name] += 1
                modality_counts[name] = modality_counts.get(name, 0) + 1
                unique_assets[str(path_value)] = name

        frame_gaps = _frame_id_gaps(sequence.frames)
        for gap in frame_gaps:
            _append_issue(
                issues,
                settings,
                severity="warning",
                code="frame_id_gap",
                sequence_id=sequence.sequence_id,
                frame_id=str(gap[0]),
                message=f"Frame id gap: {gap[0]} -> {gap[1]}.",
            )
        timestamp_issues = _timestamp_issues(sequence.frames)
        for message in timestamp_issues:
            _append_issue(
                issues,
                settings,
                severity="warning",
                code="timestamp_gap",
                sequence_id=sequence.sequence_id,
                message=message,
            )
        sequence_rows.append(
            {
                "sequence_id": sequence.sequence_id,
                "frame_count": len(sequence.frames),
                "modalities": sorted(expected_modalities),
                "asset_counts": per_sequence_counts,
                "time_start": min((frame.timestamp for frame in sequence.frames), default=None),
                "time_end": max((frame.timestamp for frame in sequence.frames), default=None),
                "frame_id_gap_count": len(frame_gaps),
                "timestamp_issue_count": len(timestamp_issues),
            }
        )

    checked_assets = 0
    corrupt_assets = 0
    for asset_ref, modality in _sample_assets(unique_assets, settings.max_asset_checks):
        checked_assets += 1
        try:
            shape = _probe_asset(asset_ref, modality)
        except Exception as exc:
            corrupt_assets += 1
            _append_issue(
                issues,
                settings,
                severity="error",
                code="corrupt_asset",
                modality=modality,
                path=asset_ref,
                message=f"Cannot read {_asset_name(asset_ref)}: {exc}",
            )
            continue
        if shape:
            modality_shapes.setdefault(modality, set()).add(shape)

    referenced_bytes = 0
    for asset_ref in unique_assets:
        try:
            referenced_bytes += _asset_size(asset_ref)
        except OSError:
            continue
    dataset_bytes, dataset_file_count, disk_usage_truncated = _directory_usage(root, settings.max_disk_files)

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "error" if error_count else "warning" if warning_count else "ready"
    asset_check_mode = "full" if checked_assets >= len(unique_assets) else "sampled"
    return {
        "dataset_root": str(root),
        "status": status,
        "training_ready": total_frames >= 2 and error_count == 0,
        "sequence_count": len(sequences),
        "sample_count": total_frames,
        "modalities": sorted(declared_modalities),
        "available_modalities": sorted(modality_counts),
        "asset_counts": modality_counts,
        "missing_asset_counts": missing_counts,
        "resolutions": {
            name: [list(shape) for shape in sorted(shapes)]
            for name, shapes in sorted(modality_shapes.items())
        },
        "time_start": min(timestamps) if timestamps else None,
        "time_end": max(timestamps) if timestamps else None,
        "duration_sec": max(timestamps) - min(timestamps) if timestamps else None,
        "referenced_disk_usage_bytes": referenced_bytes,
        "dataset_disk_usage_bytes": dataset_bytes,
        "dataset_file_count": dataset_file_count,
        "disk_usage_truncated": disk_usage_truncated,
        "checked_asset_count": checked_assets,
        "available_asset_count": len(unique_assets),
        "unchecked_asset_count": max(0, len(unique_assets) - checked_assets),
        "asset_check_mode": asset_check_mode,
        "corrupt_asset_count": corrupt_assets,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues_truncated": len(issues) >= settings.max_reported_issues,
        "issues": issues,
        "sequences": sequence_rows,
    }


def build_dataset_split(
    sequences: list[DatasetSequence],
    *,
    dataset_root: str | Path,
    adapter: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 7,
) -> dict[str, Any]:
    """Create deterministic sequence-level splits, or frame splits for one sequence."""

    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0) or not math.isclose(float(ratios.sum()), 1.0, abs_tol=1e-6):
        raise ValueError("Train, validation, and test ratios must be non-negative and sum to 1.0.")
    if not sequences:
        raise ValueError("At least one dataset sequence is required for splitting.")

    rng = np.random.default_rng(int(seed))
    split_names = ("train", "validation", "test")
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    if len(sequences) >= 3:
        order = rng.permutation(len(sequences)).tolist()
        counts = _split_counts(len(sequences), ratios)
        offset = 0
        for name, count in zip(split_names, counts, strict=True):
            for index in order[offset : offset + count]:
                sequence = sequences[index]
                split_rows[name].append(
                    {"sequence_id": sequence.sequence_id, "frame_count": len(sequence.frames)}
                )
            offset += count
        split_unit = "sequence"
        strategy = "seeded_sequence_shuffle"
        seed_applied = True
    else:
        split_unit = "frame"
        for sequence in sequences:
            order = list(range(len(sequence.frames)))
            counts = _split_counts(len(sequence.frames), ratios)
            offset = 0
            for name, count in zip(split_names, counts, strict=True):
                indices = sorted(int(index) for index in order[offset : offset + count])
                split_rows[name].append(
                    {
                        "sequence_id": sequence.sequence_id,
                        "frame_count": len(indices),
                        "frame_indices": indices,
                    }
                )
                offset += count
        strategy = "contiguous_frame_ranges"
        seed_applied = False

    return {
        "schema_version": 1,
        "dataset_root": str(Path(dataset_root).resolve()),
        "adapter": adapter,
        "seed": int(seed),
        "seed_applied": seed_applied,
        "strategy": strategy,
        "split_unit": split_unit,
        "ratios": {
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(test_ratio),
        },
        "splits": split_rows,
        "counts": {
            name: sum(int(row["frame_count"]) for row in rows)
            for name, rows in split_rows.items()
        },
    }


def _expected_modalities(sequence: DatasetSequence) -> set[str]:
    raw_expected = sequence.metadata.get("expected_modalities")
    expected = {str(name) for name in raw_expected} if isinstance(raw_expected, list) else set()
    for frame in sequence.frames:
        expected.update(frame.available_assets())
    return expected


def _sample_assets(assets: dict[str, str], limit: int) -> list[tuple[str, str]]:
    rows = sorted(assets.items(), key=lambda item: item[0].lower())
    if limit <= 0 or len(rows) <= limit:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def _probe_asset(asset_ref: str, modality: str) -> tuple[int, ...] | None:
    suffix = _asset_suffix(asset_ref)
    raw: bytes | None = None
    if asset_ref.startswith("zip://"):
        raw = _read_asset_bytes(asset_ref)
    elif not Path(asset_ref).is_file():
        raise FileNotFoundError(asset_ref)
    if suffix in IMAGE_SUFFIXES:
        source: Path | BytesIO = BytesIO(raw) if raw is not None else Path(asset_ref)
        with Image.open(source) as image:
            image.verify()
        source = BytesIO(raw) if raw is not None else Path(asset_ref)
        with Image.open(source) as image:
            return (int(image.height), int(image.width), len(image.getbands()))
    if suffix == ".npy":
        source = BytesIO(raw) if raw is not None else Path(asset_ref)
        array = np.load(source, mmap_mode=None if raw is not None else "r", allow_pickle=False)
        return tuple(int(value) for value in array.shape)
    if modality == "lidar_points" and suffix == ".bin":
        size = len(raw) if raw is not None else Path(asset_ref).stat().st_size
        if size == 0 or size % (4 * np.dtype(np.float32).itemsize) != 0:
            raise ValueError("LiDAR .bin size is not a non-empty Nx4 float32 array")
        return (size // (4 * np.dtype(np.float32).itemsize), 4)
    if (len(raw) if raw is not None else Path(asset_ref).stat().st_size) <= 0:
        raise ValueError("asset is empty")
    return None


def _asset_suffix(asset_ref: str) -> str:
    return Path(asset_ref.rsplit("!", 1)[-1]).suffix.lower()


def _asset_name(asset_ref: str) -> str:
    return Path(asset_ref.rsplit("!", 1)[-1]).name


def _asset_size(asset_ref: str) -> int:
    if not asset_ref.startswith("zip://"):
        return int(Path(asset_ref).stat().st_size)
    raw = asset_ref.removeprefix("zip://")
    zip_path, member = raw.split("!", 1)
    with zipfile.ZipFile(zip_path) as archive:
        return int(archive.getinfo(member).file_size)


def _read_asset_bytes(asset_ref: str) -> bytes:
    raw = asset_ref.removeprefix("zip://")
    zip_path, member = raw.split("!", 1)
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(member)


def _directory_usage(root: Path, limit: int) -> tuple[int, int, bool]:
    if root.is_file():
        try:
            return int(root.stat().st_size), 1, False
        except OSError:
            return 0, 0, False
    total = 0
    count = 0
    if not root.exists():
        return total, count, False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        count += 1
        try:
            total += int(path.stat().st_size)
        except OSError:
            pass
        if limit > 0 and count >= limit:
            return total, count, True
    return total, count, False


def _frame_id_gaps(frames: list[DatasetFrame]) -> list[tuple[str, str]]:
    values: list[tuple[str, int]] = []
    for frame in frames:
        match = re.search(r"(\d+)$", frame.frame_id)
        if match is None:
            return []
        values.append((frame.frame_id, int(match.group(1))))
    if len(values) < 3:
        return []
    deltas = np.diff([value for _, value in values])
    if len(deltas) == 0 or float(np.median(np.abs(deltas))) > 2.0:
        return []
    return [
        (values[index][0], values[index + 1][0])
        for index, delta in enumerate(deltas)
        if int(delta) != 1
    ]


def _timestamp_issues(frames: list[DatasetFrame]) -> list[str]:
    timestamps = np.asarray([float(frame.timestamp) for frame in frames], dtype=np.float64)
    if len(timestamps) < 2:
        return []
    deltas = np.diff(timestamps)
    issues: list[str] = []
    if np.any(deltas <= 0):
        issues.append("Timestamps are not strictly increasing.")
    positive = deltas[deltas > 0]
    if len(positive) >= 3:
        median = float(np.median(positive))
        if median > 0 and np.any(positive > median * 5.0):
            issues.append("Timestamp interval contains a gap larger than 5x the median interval.")
    return issues


def _split_counts(total: int, ratios: np.ndarray) -> tuple[int, int, int]:
    raw = ratios * int(total)
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts))[: int(total - int(counts.sum()))]:
        counts[int(index)] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _append_issue(
    issues: list[dict[str, Any]],
    options: DatasetAnalysisOptions,
    *,
    severity: str,
    code: str,
    message: str,
    sequence_id: str = "",
    frame_id: str = "",
    modality: str = "",
    path: str = "",
) -> None:
    if len(issues) >= options.max_reported_issues:
        return
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "modality": modality,
            "path": path,
        }
    )
