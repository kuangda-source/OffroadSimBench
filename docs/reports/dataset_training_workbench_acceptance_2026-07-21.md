# Dataset And Training Workbench Acceptance

Date: 2026-07-21

## Scope

- Dataset inspection, synchronized preview, quality state, and deterministic split.
- Manifest mapping for an unrelated directory layout.
- Schema-backed built-in and external trainers.
- Live metrics, resource monitoring, resume, and checkpoint management.
- Independent checkpoint inference and error-sample review.
- Experiment comparison and versioned non-destructive processing.

## Real ORFD Result

Dataset root: `datasets/ORFD_Dataset_ICRA2022_ZIP`

- Adapter: `orfd`
- Detected sequences: 30
- Selected sequence: `training/c2021_0228_1819`
- Selected frames: 449
- Modalities: RGB, depth, label, LiDAR
- Quality: ready; sampled corrupt-asset count 0
- Split counts: train 7872, validation 2639, test 1687
- Tiny RGB Depth parameters: 6 frames, 2 epochs, 64 pixels/frame
- Training status: completed
- Validation loss: 2.7735606315
- Inference status: completed
- Test samples: 3
- Depth RMSE: 5.2099332809 m
- Depth MAE: 3.3855688572 m

This is a workflow acceptance run, not a model-accuracy claim.

## Extensibility Result

A second eight-frame RGB/depth dataset used unrelated paths (`camera_left` and
`range_groundtruth`) and filename prefixes. A saved `manifest_dataset` mapping
aligned files by frame-id regex, then completed split, Tiny RGB Depth training,
and test inference without adding or changing a dataset adapter.

An external Python trainer is also covered through entrypoint path, parameter
schema, command template, reusable training config, and subprocess execution.

## Automated Validation

The acceptance paths are covered by:

- `tests/test_training_workbench_e2e.py`
- `tests/test_inference_workbench.py`
- `tests/test_training_manifest_services.py`
- `tests/test_desktop_services.py`
- `tests/test_desktop_visible_demo.py`

The project-wide validation commands remain those documented in `AGENTS.md`.
