"""Train a tiny RGB-to-depth regression baseline from a registered dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from offroad_sim.datasets import default_dataset_registry
from offroad_sim.training.tiny_depth import TinyDepthModel, dataset_split_frames, sample_frame_pixels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sequence-id", default="")
    parser.add_argument("--split-path", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--max-pixels-per-frame", type=int, default=512)
    parser.add_argument("--max-depth-m", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = default_dataset_registry()
    adapter = registry.resolve(args.dataset_root, args.adapter)
    sequence_id = args.sequence_id or adapter.list_sequences(args.dataset_root)[0]
    if args.split_path:
        train_frames = dataset_split_frames(adapter, args.dataset_root, args.split_path, "train")
        validation_frames = dataset_split_frames(adapter, args.dataset_root, args.split_path, "validation")
        train_frames = [frame for frame in train_frames if frame.front_rgb_path and frame.depth_path][: args.max_frames]
        validation_frames = [
            frame for frame in validation_frames if frame.front_rgb_path and frame.depth_path
        ][: args.max_frames]
    else:
        sequence = adapter.load_sequence(args.dataset_root, sequence_id)
        frames = [frame for frame in sequence.frames if frame.front_rgb_path and frame.depth_path][: args.max_frames]
        if len(frames) < 2:
            raise ValueError("Tiny depth training requires at least two synchronized RGB/depth frames.")
        split_index = max(1, min(len(frames) - 1, int(round(len(frames) * 0.75))))
        train_frames = frames[:split_index]
        validation_frames = frames[split_index:]
    if not train_frames or not validation_frames:
        raise ValueError("Tiny depth training requires non-empty train and validation RGB/depth splits.")
    rng = np.random.default_rng(args.seed)

    def samples(selected_frames):
        rows = [
            sample_frame_pixels(
                frame,
                max_pixels=args.max_pixels_per_frame,
                rng=rng,
                max_depth_m=args.max_depth_m,
            )
            for frame in selected_frames
        ]
        return np.vstack([row[0] for row in rows]), np.concatenate([row[1] for row in rows])

    train_x, train_y = samples(train_frames)
    validation_x, validation_y = samples(validation_frames)
    if not all(np.all(np.isfinite(value)) for value in (train_x, train_y, validation_x, validation_y)):
        raise ValueError("Tiny depth training data contains NaN or infinite values.")
    start_epoch = 0
    if args.resume:
        resumed_model = TinyDepthModel.load(args.resume)
        weights = resumed_model.weights.astype(np.float64, copy=True)
        if len(weights) != train_x.shape[1]:
            raise ValueError(
                f"Resume checkpoint has {len(weights)} weights, expected {train_x.shape[1]}."
            )
        start_epoch = int(resumed_model.metadata.get("epoch") or 0)
    else:
        weights = np.zeros(train_x.shape[1], dtype=np.float64)
        weights[0] = float(np.mean(train_y))
    history = {"train_loss": [], "validation_loss": [], "learning_rate": [], "throughput": []}
    events_path = output_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as events:
        for local_epoch in range(1, args.epochs + 1):
            epoch = start_epoch + local_epoch
            with np.errstate(over="ignore", invalid="ignore"):
                train_error = train_x @ weights - train_y
                regularization = weights.copy()
                regularization[0] = 0.0
                gradient = (2.0 / len(train_y)) * (train_x.T @ train_error) + 2.0 * args.ridge * regularization
                next_weights = weights - args.learning_rate * gradient
                train_rmse = float(np.sqrt(np.mean(np.square(train_x @ next_weights - train_y))))
                validation_rmse = float(np.sqrt(np.mean(np.square(validation_x @ next_weights - validation_y))))
            if not (
                np.all(np.isfinite(gradient))
                and np.all(np.isfinite(next_weights))
                and np.isfinite(train_rmse)
                and np.isfinite(validation_rmse)
            ):
                raise FloatingPointError(
                    f"Tiny depth training diverged at epoch {epoch}; reduce learning_rate or normalize inputs."
                )
            weights = next_weights
            event = {
                "epoch": epoch,
                "step": epoch,
                "train_loss": train_rmse,
                "validation_loss": validation_rmse,
                "learning_rate": args.learning_rate,
                "throughput": float(len(train_y)),
                "progress": local_epoch / args.epochs,
                "current_step": local_epoch,
                "total_steps": args.epochs,
                "message": f"depth epoch {epoch} (+{local_epoch}/{args.epochs})",
            }
            events.write(json.dumps(event) + "\n")
            events.flush()
            print(json.dumps(event), flush=True)
            for key in history:
                history[key].append(float(event[key]))

    metrics = {
        "train_rmse_m": history["train_loss"][-1],
        "validation_rmse_m": history["validation_loss"][-1],
        "epoch": start_epoch + args.epochs,
        "train_frame_count": len(train_frames),
        "validation_frame_count": len(validation_frames),
        "train_pixel_count": len(train_y),
        "validation_pixel_count": len(validation_y),
    }
    model = TinyDepthModel(
        weights=weights,
        max_depth_m=args.max_depth_m,
        metadata={
            **metrics,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "adapter": adapter.name,
            "sequence_id": sequence_id,
            "split_path": str(Path(args.split_path).resolve()) if args.split_path else "",
            "ridge": args.ridge,
            "learning_rate": args.learning_rate,
            "resumed_from": str(Path(args.resume).resolve()) if args.resume else "",
        },
    )
    model_path = model.save(output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    payload = {
        "artifact_path": str(model_path.resolve()),
        "artifact_type": "depth_model",
        "model_path": str(model_path.resolve()),
        "output_dir": str(output_dir),
        "metrics": metrics,
        "history": history,
        "split_path": str(Path(args.split_path).resolve()) if args.split_path else "",
    }
    print(json.dumps(payload, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
