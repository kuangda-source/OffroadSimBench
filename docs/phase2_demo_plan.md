# Phase 2 Demo Plan

Phase 2 turns the first-stage benchmark skeleton into a visible local demo.
The demo target is an operator-style dashboard that can stream a heightmap
episode, inspect terrain risk and local BEV frames, replay recorded runs, and
show whether the local BeamNG runtime is ready.

## Demo Scope

- Run a `gym_heightmap` episode from the dashboard through the SSE endpoint.
- Visualize the global terrain-risk layer with an overlaid vehicle trajectory.
- Visualize the current local BEV risk patch.
- Track key metrics during the run: reward, distance, speed, risk, collision,
  path length, and completion state.
- Load a recorded episode and replay it with play/pause, step, reset, and a
  timeline slider.
- Report BeamNG adapter readiness from the same backend registry used by the
  API and CLI.

## Local Startup

```powershell
$env:MAMBA_ROOT_PREFIX = "D:\programs\OffroadSimBench\.mamba-root"
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" run -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd D:\programs\OffroadSimBench\dashboard\frontend
npm run dev
```

Open `http://127.0.0.1:5173`.

## Acceptance Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase2_acceptance.ps1
```

To also launch and connect to BeamNG through `beamngpy`, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase2_acceptance.ps1 -BeamNGConnect
```

The default acceptance path verifies BeamNG readiness without launching the
simulator. That keeps the automated check fast and repeatable while leaving a
real connection smoke test available for local machine validation.
