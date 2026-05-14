# Phase 2 Acceptance Report

Date: 2026-05-14

## Scope

Phase 2 is accepted as a visible local demo milestone. The project now supports
live dashboard episode streaming, recorded replay inspection, BeamNG runtime
readiness checks, and an optional real BeamNG connection smoke test.

## Delivered

- Added `/stream_episode` SSE API for live `gym_heightmap` runs.
- Added `/episodes/{episode_id}/steps` for full recorded episode replay.
- Added `/beamng/status` and `examples/check_beamng_runtime.py`.
- Added dashboard views for terrain risk, local BEV risk, trajectory overlay,
  metrics, timeline replay, and recent episode loading.
- Added BeamNG auto-detection for local `BeamNG/BeamNG.tech*` installs.
- Added `beamngpy` to the project environment and optional Python extra.
- Added `scripts/phase2_acceptance.ps1` and `docs/phase2_demo_plan.md`.

## Validation Results

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase2_acceptance.ps1 -BeamNGConnect
```

Result:

- Python tests: `41 passed, 1 skipped`
- Backend catalog: `beamng`, `dataset_replay`, `gym_heightmap`, and `ue5` available
- BeamNG runtime status: ready
- CLI recorded smoke episode: passed
- Dashboard API streaming smoke: `5 passed`
- Frontend production build: passed
- BeamNG real connection smoke: reset plus one control step passed

BeamNG runtime details:

```text
bng_home: D:\programs\OffroadSimBench\BeamNG\BeamNG.tech.v0.38.3.0
executable: D:\programs\OffroadSimBench\BeamNG\BeamNG.tech.v0.38.3.0\BeamNG.tech.exe
level: west_coast_usa
vehicle_id: ego
episode_length: 1
```

## Browser Demo Check

The dashboard was opened at `http://127.0.0.1:5173` with the API at
`http://127.0.0.1:8000`.

Observed:

- API health displayed `ok`.
- BeamNG status displayed `ready`.
- A streamed `forest_trail_001_rule_based` run produced `129` frames.
- Terrain risk map, local BEV map, trajectory overlay, replay slider, and
  metrics table rendered without visible overlap.
- Streamed metrics included `success=true`, `steps=128`, `collision_count=0`,
  and `distance_to_goal=4.528`.

## Acceptance Decision

Phase 2 is accepted. The visible demonstration target is met, and BeamNG is
confirmed through both runtime status and a minimal real connection smoke test.

## Known Limits

- Live dashboard streaming is currently limited to `gym_heightmap`; BeamNG is
  connected through the backend adapter and smoke script.
- BeamNG camera/lidar/IMU array parsing is not implemented yet.
- BeamNG scenario placement still uses the minimal default vehicle spawn path;
  scenario-specific BeamNG maps and spawn transforms should be added next.
