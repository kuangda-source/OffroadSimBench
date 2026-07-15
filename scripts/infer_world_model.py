"""Run one-step TinyLearnedWorldModel inference on a registered dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from offroad_sim.core import Action, Observation
from offroad_sim.datasets import default_dataset_registry
from offroad_sim.world_models import TinyLearnedWorldModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_path")
    parser.add_argument("dataset_root")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sequence-id", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = TinyLearnedWorldModel.load(args.artifact_path)
    registry = default_dataset_registry()
    adapter = registry.resolve(args.dataset_root, args.adapter)
    sequence_id = args.sequence_id or adapter.list_sequences(args.dataset_root)[0]
    sequence = adapter.load_sequence(args.dataset_root, sequence_id)

    predictions: list[dict[str, object]] = []
    errors: list[list[float]] = []
    predicted_xy: list[tuple[float, float]] = []
    actual_xy: list[tuple[float, float]] = []
    transitions = list(zip(sequence.frames, sequence.frames[1:]))[: max(1, args.max_samples)]
    for index, (current, target) in enumerate(transitions):
        action = current.action or _inferred_action(current.vehicle_state, target.vehicle_state)
        goal = sequence.goal or (target.vehicle_state.x, target.vehicle_state.y)
        observation = Observation(
            timestamp=current.timestamp,
            vehicle_state=current.vehicle_state,
            goal=goal,
        )
        predicted = model.predict(observation, action, horizon=1).states[0]
        actual = target.vehicle_state
        yaw_error = _angle_delta(predicted.yaw, actual.yaw)
        errors.append(
            [
                predicted.x - actual.x,
                predicted.y - actual.y,
                yaw_error,
                predicted.speed - actual.speed,
            ]
        )
        predicted_xy.append((predicted.x, predicted.y))
        actual_xy.append((actual.x, actual.y))
        predictions.append(
            {
                "sample": index,
                "frame_id": current.frame_id,
                "target_frame_id": target.frame_id,
                "predicted": {
                    "x": predicted.x,
                    "y": predicted.y,
                    "yaw": predicted.yaw,
                    "speed": predicted.speed,
                },
                "actual": {
                    "x": actual.x,
                    "y": actual.y,
                    "yaw": actual.yaw,
                    "speed": actual.speed,
                },
            }
        )
    if not errors:
        raise ValueError("Inference requires a sequence with at least two frames.")

    error_array = np.asarray(errors, dtype=np.float64)
    metrics = {
        "sample_count": len(predictions),
        "position_rmse": float(np.sqrt(np.mean(error_array[:, :2] ** 2))),
        "yaw_rmse": float(np.sqrt(np.mean(error_array[:, 2] ** 2))),
        "speed_rmse": float(np.sqrt(np.mean(error_array[:, 3] ** 2))),
    }
    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preview_path = output_dir / "trajectory_comparison.png"
    _draw_trajectory_preview(actual_xy, predicted_xy, preview_path)
    print(
        json.dumps(
            {
                "metrics": metrics,
                "predictions": predictions,
                "previews": {"trajectory": str(preview_path)},
            },
            default=str,
        )
    )
    return 0


def _inferred_action(current, target) -> Action:
    yaw_delta = _angle_delta(target.yaw, current.yaw)
    speed_delta = target.speed - current.speed
    return Action(
        steer=max(-1.0, min(1.0, yaw_delta)),
        throttle=max(0.0, min(1.0, speed_delta)),
        brake=max(0.0, min(1.0, -speed_delta)),
        gear=1,
    )


def _angle_delta(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


def _draw_trajectory_preview(
    actual: list[tuple[float, float]],
    predicted: list[tuple[float, float]],
    path: Path,
) -> None:
    width, height, margin = 900, 520, 40
    image = Image.new("RGB", (width, height), (250, 250, 252))
    draw = ImageDraw.Draw(image)
    points = actual + predicted
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)

    def project(point: tuple[float, float]) -> tuple[int, int]:
        x = margin + int((point[0] - x_min) / x_span * (width - 2 * margin))
        y = height - margin - int((point[1] - y_min) / y_span * (height - 2 * margin))
        return x, y

    actual_pixels = [project(point) for point in actual]
    predicted_pixels = [project(point) for point in predicted]
    if len(actual_pixels) >= 2:
        draw.line(actual_pixels, fill=(0, 122, 255), width=4)
    if len(predicted_pixels) >= 2:
        draw.line(predicted_pixels, fill=(255, 149, 0), width=4)
    draw.text((margin, 14), "Actual", fill=(0, 122, 255))
    draw.text((margin + 90, 14), "Predicted", fill=(255, 149, 0))
    image.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
