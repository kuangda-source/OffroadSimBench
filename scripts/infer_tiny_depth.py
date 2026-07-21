"""Run a tiny RGB-to-depth checkpoint on synchronized dataset frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from offroad_sim.datasets import default_dataset_registry
from offroad_sim.training.tiny_depth import TinyDepthModel, dataset_split_frames, depth_colormap, frame_depth_arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_path")
    parser.add_argument("dataset_root")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sequence-id", default="")
    parser.add_argument("--split-path", default="")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = TinyDepthModel.load(args.artifact_path)
    registry = default_dataset_registry()
    adapter = registry.resolve(args.dataset_root, args.adapter)
    sequence_id = args.sequence_id or adapter.list_sequences(args.dataset_root)[0]
    if args.split_path:
        frames = dataset_split_frames(adapter, args.dataset_root, args.split_path, args.split_name)
        locations = _split_frame_locations(adapter, args.dataset_root, args.split_path, args.split_name)
    else:
        sequence_frames = adapter.load_sequence(args.dataset_root, sequence_id).frames
        frames = sequence_frames
        locations = [(sequence_id, index) for index in range(len(sequence_frames))]
    samples = [
        (frame, location)
        for frame, location in zip(frames, locations)
        if frame.front_rgb_path and frame.depth_path
    ][: args.max_samples]
    if not samples:
        raise ValueError("No synchronized RGB/depth frames are available for inference.")

    predictions = []
    preview_path = output_dir / "depth_comparison.png"
    sample_preview_dir = output_dir / "sample_previews"
    sample_preview_dir.mkdir(parents=True, exist_ok=True)
    for index, (frame, location) in enumerate(samples):
        frame_sequence_id, frame_index = location
        rgb, target = frame_depth_arrays(frame)
        prediction = model.predict(rgb)
        valid = np.isfinite(target) & (target > 0.0) & (target <= model.max_depth_m)
        error = prediction[valid] - target[valid]
        stem = f"{index:05d}_{_safe_name(frame.frame_id)}"
        input_path = sample_preview_dir / f"{stem}_input.png"
        target_path = sample_preview_dir / f"{stem}_target.png"
        prediction_path = sample_preview_dir / f"{stem}_prediction.png"
        error_path = sample_preview_dir / f"{stem}_error.png"
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(input_path)
        Image.fromarray(depth_colormap(target, max_depth_m=model.max_depth_m)).save(target_path)
        Image.fromarray(depth_colormap(prediction, max_depth_m=model.max_depth_m)).save(prediction_path)
        absolute_error = np.where(valid, np.abs(prediction - target), 0.0)
        Image.fromarray(depth_colormap(absolute_error, max_depth_m=max(1.0, model.max_depth_m * 0.25))).save(
            error_path
        )
        predictions.append(
            {
                "sample": index,
                "sequence_id": frame_sequence_id,
                "frame_index": frame_index,
                "frame_id": frame.frame_id,
                "rmse_m": float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan"),
                "mae_m": float(np.mean(np.abs(error))) if error.size else float("nan"),
                "valid_pixel_count": int(error.size),
                "previews": {
                    "input": str(input_path.resolve()),
                    "target": str(target_path.resolve()),
                    "prediction": str(prediction_path.resolve()),
                    "error": str(error_path.resolve()),
                },
            }
        )
        if index == 0:
            preview = np.concatenate(
                [
                    rgb[..., :3].astype(np.uint8),
                    depth_colormap(target, max_depth_m=model.max_depth_m),
                    depth_colormap(prediction, max_depth_m=model.max_depth_m),
                ],
                axis=1,
            )
            image = Image.fromarray(preview)
            if image.width > 1440:
                height = max(1, int(round(image.height * 1440 / image.width)))
                image = image.resize((1440, height), Image.Resampling.BILINEAR)
            image.save(preview_path)

    finite_rmse = [row["rmse_m"] for row in predictions if np.isfinite(row["rmse_m"])]
    finite_mae = [row["mae_m"] for row in predictions if np.isfinite(row["mae_m"])]
    metrics = {
        "depth_rmse_m": float(np.mean(finite_rmse)) if finite_rmse else float("nan"),
        "depth_mae_m": float(np.mean(finite_mae)) if finite_mae else float("nan"),
        "sample_count": len(predictions),
        "split_name": args.split_name if args.split_path else "selected_sequence",
    }
    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": metrics,
                "predictions": predictions,
                "preview_path": str(preview_path.resolve()),
                "predictions_path": str(predictions_path.resolve()),
            }
        ),
        flush=True,
    )
    return 0


def _safe_name(value: object) -> str:
    text = str(value or "frame")
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in text)


def _split_frame_locations(
    adapter: object,
    dataset_root: str,
    split_path: str,
    split_name: str,
) -> list[tuple[str, int]]:
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    splits = payload.get("splits") if isinstance(payload.get("splits"), dict) else {}
    rows = splits.get(split_name) if isinstance(splits.get(split_name), list) else []
    locations: list[tuple[str, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sequence_id = str(row.get("sequence_id") or "")
        indices = row.get("frame_indices")
        if isinstance(indices, list):
            locations.extend((sequence_id, int(index)) for index in indices)
        else:
            sequence = adapter.load_sequence(dataset_root, sequence_id)  # type: ignore[attr-defined]
            locations.extend((sequence_id, index) for index in range(len(sequence.frames)))
    return locations


if __name__ == "__main__":
    raise SystemExit(main())
