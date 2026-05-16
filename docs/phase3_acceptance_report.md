# Phase 3 Acceptance Report

Date: 2026-05-15

## Scope

Phase 3 is accepted as the first real-data/world-model/BeamNG integration
milestone. The project now has switchable agent and world-model registries, an
ORFD dataset adapter, a small learned world model trained from dataset
sequences, an optional LE-WM wrapper boundary, and a BeamNG world-model episode
smoke path.

## Delivered

- Added `AgentRegistry` so driving algorithms can be selected without
  application-level hard-coding.
- Added `WorldModelRegistry` with `simple_kinematic`, `tiny_learned`, and
  optional `le_wm` entries.
- Added `TinyLearnedWorldModel`, a NumPy linear dynamics model that trains from
  registered dataset sequences and saves/loads from disk.
- Added `LeWMWorldModel` and `le_wm_cem` as the stable-worldmodel
  checkpoint/runtime wrapper for `https://github.com/lucas-maes/le-wm`.
- Added `ORFDAdapter` for ORFD-style `training|validation|testing/<sequence>`
  folders with `image_data`, depth, lidar, ground-truth, and calibration assets.
- Added dataset inspection, mock ORFD fixture generation, tiny-model training,
  LE-WM HDF5 export boundary, and BeamNG world-model run scripts.
- Generalized `run_episode()` to registered backends beyond `gym_heightmap`.
- Added PySide6 desktop GUI controls for world model type/path, dataset
  root/sequence/adapter selection, HDF5 export, and LE-WM cost-model training.
- Updated the GitHub README into Chinese/English sections with top language
  switch links.

## Validation Results

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase3_acceptance.ps1 -BeamNGConnect
```

Result:

- Python tests: `55 passed, 1 skipped`
- ORFD inspection: adapter `orfd`, sequence `training/seq_0001`, `8` frames
- ORFD asset counts: `front_rgb=8`, `depth=8`, `lidar_points=8`, `label=8`
- Tiny learned model training:
  - model type: `tiny_learned`
  - samples: `7`
  - train RMSE: `1.974688044705451e-05`
- Dataset replay with world-model agent:
  - backend: `dataset_replay`
  - agent: `world_model`
  - world model: `tiny_learned`
  - steps: `5`
  - episode recorded under `outputs/episodes`
- StableWM HDF5 export: `stable_worldmodel_flat_v1`
- LE-WM-compatible cost checkpoint: `_object.ckpt` loadable by `AutoCostModel`
- Desktop GUI offscreen smoke: passed
- BeamNG runtime status: `ready`
- BeamNG LE-WM CEM connection run:
  - scenario: `beamng_orfd_eval`
  - backend: `beamng`
  - world model: `le_wm`
  - planner: `le_wm_cem`
  - steps: `3`
  - connected: `true`
  - episode recorded under `outputs/episodes`

## Acceptance Decision

Phase 3 is accepted. The required chain is now present:

```text
ORFD/BeamNG data -> StableWM HDF5 -> LE-WM-compatible checkpoint -> le_wm_cem -> BeamNG run -> recorded metrics/replay
```

The implementation keeps model and algorithm switching behind registries, so a
future LE-WM checkpoint or another learned model can be integrated without
hard-coded changes in the runner, CLI, or desktop GUI controls.

## Known Limits

- ORFD is a freespace/perception dataset and may not include vehicle actions or
  ego poses. The adapter uses `poses.csv` when available; otherwise it marks
  poses as `synthetic_index_order`.
- The current LE-WM checkpoint is a small stable-worldmodel cost-model smoke
  checkpoint. It validates training/loading/planning but is not a full upstream
  LE-WM research model.
- The local BeamNG run succeeded, but `sensor_count=0` for the current vehicle
  path because no vehicle config was passed into the BeamNG episode runner.
  Sensor attachment support is in place for later vehicle-config-driven runs.
