# Phase 4 Acceptance Report

Date: 2026-05-16

## Scope

Phase 4 is accepted when OffroadSimBench can launch BeamNG visibly and drive a
vehicle with the normal dataset/model/planner selection path. The accepted demo
uses the real ORFD dataset selection and the local LE-WM-compatible checkpoint
trained from that data, then drives a BeamNG vehicle through `route_world_model`
and `le_wm_cem`.

## Delivered

- Added `configs/scenarios/beamng_visible_autodrive.yaml`.
- Added BeamNG scenario metadata handling for visible level, spawn pose, route,
  route markers, camera setup, and step cadence.
- Added BeamNG motion metrics including `distance_traveled`,
  `route_waypoint_count`, `collision_count`, and active `level`.
- Added `route_world_model`, a waypoint-following agent that keeps world model
  and planner switching behind registries.
- Added visible BeamNG demo service, CLI script, GUI button, and focused tests.
- Added `scripts/phase4_visible_beamng_acceptance.ps1`.

## Validation Commands

Focused tests:

```powershell
python -m pytest tests/test_beamng_visible_config.py tests/test_beamng_backend_visible.py tests/test_route_world_model_agent.py tests/test_desktop_visible_demo.py -q
```

Visible BeamNG LE-WM run:

```powershell
python scripts\run_beamng_visible_demo.py `
  --dataset-root datasets\ORFD_Dataset_ICRA2022_ZIP `
  --adapter orfd `
  --sequence-id training/c2021_0228_1819 `
  --world-model-type le_wm `
  --world-model outputs\models\lewm_orfd_real_c2021_0228_1819 `
  --planner le_wm_cem `
  --scenario beamng_visible_autodrive `
  --vehicle configs\vehicles\ugv_medium.yaml `
  --max-steps 80
```

## Local Acceptance Result

- BeamNG connected: `true`
- BeamNG level: `gridmap_v2`
- Agent: `route_world_model`
- World model: `le_wm`
- Planner: `le_wm_cem`
- Steps: `80`
- Route waypoints: `4`
- Distance traveled: `238.49808425341425m`
- Episode: `beamng_visible_autodrive_route_world_model_20260516T143734Z`
- Episode path:
  `D:\programs\OffroadSimBench\outputs\episodes\beamng_visible_autodrive_route_world_model_20260516T143734Z`

## Known Limits

- The current accepted visual demo uses a stock BeamNG level. Full ORFD terrain
  reconstruction and packaging remains a later map-building task.
- BeamNG sensor attachment is still best-effort across beamngpy versions. The
  visible driving loop relies on BeamNG vehicle pose and dynamics.
- The LE-WM checkpoint used here is the local stable-worldmodel-compatible cost
  checkpoint, not full upstream latent-video LE-WM inference.
