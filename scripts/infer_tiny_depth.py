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
    else:
        frames = adapter.load_sequence(args.dataset_root, sequence_id).frames
    frames = [frame for frame in frames if frame.front_rgb_path and frame.depth_path][: args.max_samples]
    if not frames:
        raise ValueError("No synchronized RGB/depth frames are available for inference.")

    predictions = []
    preview_path = output_dir / "depth_comparison.png"
    for index, frame in enumerate(frames):
        rgb, target = frame_depth_arrays(frame)
        prediction = model.predict(rgb)
        valid = np.isfinite(target) & (target > 0.0) & (target <= model.max_depth_m)
        error = prediction[valid] - target[valid]
        predictions.append(
            {
                "frame_id": frame.frame_id,
                "rmse_m": float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan"),
                "mae_m": float(np.mean(np.abs(error))) if error.size else float("nan"),
                "valid_pixel_count": int(error.size),
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


if __name__ == "__main__":
    raise SystemExit(main())
