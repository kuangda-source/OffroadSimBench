# Phase 4 Development Plan

Date: 2026-05-16

## Goal

Phase 4 turns the Phase 3 BeamNG smoke path into a visible autonomous-driving
demo. The target loop is:

```text
ORFD or another dataset -> selected world model/checkpoint -> selected planner -> route_world_model agent -> visible BeamNG vehicle -> recorded episode
```

## Deliverables

- `beamng_visible_autodrive` scenario for a stable visible demo on a stock
  BeamNG level.
- BeamNG backend support for scenario-defined vehicle model, spawn pose, route
  metadata, route markers, camera setup, and motion metrics.
- `route_world_model` agent that follows route waypoints while preserving the
  existing switchable world-model and planner interfaces.
- CLI entrypoint: `scripts/run_beamng_visible_demo.py`.
- Desktop GUI BeamNG action: `启动 BeamNG 可视自动驾驶`.
- Repeatable acceptance script:
  `scripts/phase4_visible_beamng_acceptance.ps1`.

## Acceptance

Non-launching checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1
```

Visible BeamNG run using the local ORFD-derived LE-WM-compatible checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1 `
  -BeamNGVisible `
  -OrfdRoot datasets\ORFD_Dataset_ICRA2022_ZIP `
  -SequenceId training/c2021_0228_1819 `
  -WorldModelType le_wm `
  -WorldModelPath outputs\models\lewm_orfd_real_c2021_0228_1819 `
  -Planner le_wm_cem `
  -MaxSteps 80
```

The visible run is accepted when BeamNG reports `connected=true`, the episode
runs at least 60 steps, the vehicle travels at least 10 meters, and the episode
directory is recorded under `outputs/episodes`.

## Boundary

This phase does not claim full ORFD scene reconstruction as a BeamNG level.
ORFD is used as a real data source for model export/training and model
selection, while BeamNG uses a stock drivable level for the visible closed-loop
vehicle demonstration.
