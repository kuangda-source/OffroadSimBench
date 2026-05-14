# Phase 3 Development Plan

Date: 2026-05-15

## Goal

Phase 3 targets a real-data and real-simulator loop:

1. read a small ORFD dataset split through a dataset adapter;
2. train or load a switchable world model;
3. run a world-model agent through the same runner/API used by other agents;
4. optionally launch BeamNG and record a short world-model-controlled episode;
5. show dataset, model, backend, replay, and metrics from the dashboard.

## Delivered Interfaces

- `ORFDAdapter` reads ORFD-style `training|validation|testing/<sequence>/image_data`,
  `dense_depth`, `sparse_depth`, `lidar_data`, `gt_image`, and `calib` layouts.
- `AgentRegistry` makes driving algorithms switchable without hard-coded
  application branches.
- `WorldModelRegistry` exposes `simple_kinematic`, `tiny_learned`, and optional
  `le_wm` model entries.
- `TinyLearnedWorldModel` provides a small NumPy dynamics model for real-data
  smoke tests and reproducible acceptance.
- `LeWMWorldModel` reserves the external LE-WM checkpoint/runtime boundary
  without vendoring the upstream repository.
- `run_episode()` and dashboard streaming now accept registered backends beyond
  `gym_heightmap`.

## Acceptance

Default smoke acceptance:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1
```

With a local ORFD root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -OrfdRoot D:\datasets\ORFD
```

With BeamNG launch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -OrfdRoot D:\datasets\ORFD -BeamNGConnect
```

## Known Limits

- ORFD does not always provide vehicle actions or ego poses. The adapter uses
  `poses.csv` when present; otherwise it marks frame poses as
  `synthetic_index_order`.
- LE-WM is represented as an optional wrapper. Full inference requires installing
  the upstream runtime and adding checkpoint-specific input conversion in
  `offroad_sim/world_models/le_wm.py`.
- BeamNG sensor support is best-effort across beamngpy versions. Pose and
  vehicle state are used when available; camera/lidar arrays are attached and
  polled when supported by the installed runtime.
