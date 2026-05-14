# OffroadSimBench

OffroadSimBench is a local off-road autonomous driving simulation and world-model evaluation platform.

The first-stage milestone now provides a runnable local benchmark foundation:

- a modular Python package named `offroad_sim`;
- shared core types for actions, observations, vehicle state, and step results;
- base interfaces for simulator backends and driving agents;
- vehicle and scenario configuration loading from YAML;
- lightweight Gym/heightmap simulation, dataset replay, BeamNG/UE5 adapters, evaluation, replay, world-model and RL wrappers, a CLI, and a local dashboard.

The second-stage milestone adds the first visible demo path:

- a dashboard SSE stream for live episode execution;
- terrain-risk, local BEV, trajectory, metric, and timeline views;
- recorded episode replay with play/pause and frame stepping;
- BeamNG runtime auto-detection and status checks.

## Project Direction

This is not intended to be only a web demo. The dashboard should eventually act as a local control panel, while simulation backends do the real work of running vehicles, agents, metrics, and replay.

The core design principle is:

```text
Agent code depends on the OffroadAgent API.
Simulator code depends on the OffroadSimBackend API.
BeamNG, UE5, Gym, and Dataset Replay stay behind backend adapters.
```

## Current Skeleton

```text
offroad_sim/
  agents/
    base.py
    basic.py
  backends/
    base.py
    beamng_backend.py
    dataset_replay_backend.py
    gym_heightmap_backend.py
    registry.py
    ue5_backend.py
  core/
    types.py
  datasets/
    adapters/
    mock.py
    registry.py
    types.py
  evaluation/
    metrics.py
    runner.py
  replay/
    episode.py
  rl/
    gymnasium_env.py
  scenarios/
  utils/
  vehicles/
  world_models/
    base.py
    kinematic.py
configs/
  scenarios/
    forest_trail_001.yaml
  vehicles/
    ugv_medium.yaml
dashboard/
  backend/
  frontend/
docs/
  beamng_backend.md
  ue5_backend.md
examples/
  check_backends.py
  run_gym_demo.py
  run_dataset_replay.py
  run_mock_ue5_backend.py
  replay_episode.py
scripts/
  create_mock_dataset.py
tests/
```

## Setup

Create and activate a Python environment, then install the project in editable mode:

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

### Conda Environment

This repo also includes `environment.yml` for a local conda-style environment. On this machine, the environment was created with the bundled micromamba executable:

```powershell
$env:MAMBA_ROOT_PREFIX = "D:\programs\OffroadSimBench\.mamba-root"
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" create -y -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" -f environment.yml
```

Run commands inside it with:

```powershell
$env:MAMBA_ROOT_PREFIX = "D:\programs\OffroadSimBench\.mamba-root"
& "D:\programs\OffroadSimBench\BeamNG\tools\Library\bin\micromamba.exe" run -p "D:\programs\OffroadSimBench\.conda\offroad-sim-bench" python -m pytest
```

## Validation

Run the initial test suite:

```bash
pytest
```

Run a local episode from the CLI:

```bash
offroad-sim run --agent rule_based --max-steps 1200
```

Check package import directly:

```bash
python -c "import offroad_sim; print(offroad_sim.__version__)"
```

## Core Interfaces

The first shared API layer is now in place:

- `offroad_sim.core.Action`
- `offroad_sim.core.VehicleState`
- `offroad_sim.core.Observation`
- `offroad_sim.core.StepResult`
- `offroad_sim.core.EpisodeInfo`
- `offroad_sim.agents.OffroadAgent`
- `offroad_sim.backends.OffroadSimBackend`

All future agents should implement `OffroadAgent`. All simulator integrations should implement `OffroadSimBackend`.

## Backend Registry

M9/M10 add a runtime backend registry so application code can switch simulators without importing simulator-specific modules directly:

```python
from offroad_sim.backends import default_backend_registry, make_backend

registry = default_backend_registry()
print(registry.names())
backend = make_backend("gym_heightmap")
```

Current backend names are:

- `gym_heightmap`
- `dataset_replay`
- `beamng`
- `ue5`

Check local backend availability:

```bash
python examples/check_backends.py
```

## CLI

The `offroad-sim` command exposes the first-stage local workflows:

```bash
offroad-sim list
offroad-sim run --agent world_model --max-steps 500 --record
offroad-sim replay outputs/episodes/<episode_id>
```

The local automated runner currently executes `gym_heightmap` episodes. BeamNG and UE5 stay available through their backend adapters and status checks because they require external simulator runtimes.

## Config Loading

Vehicle configs live under `configs/vehicles/`:

```python
from offroad_sim.vehicles import load_vehicle_config

vehicle = load_vehicle_config("configs/vehicles/ugv_medium.yaml")
```

Scenario configs live under `configs/scenarios/`:

```python
from offroad_sim.scenarios import load_scenario_config

scenario = load_scenario_config("configs/scenarios/forest_trail_001.yaml")
```

## Gym Heightmap Backend

The lightweight `GymHeightmapBackend` supports fast local experiments without BeamNG or UE5. It generates a deterministic 2.5D terrain map with:

- heightmap;
- occupancy map;
- traversability map;
- terrain risk map.

Run a demo episode with the rule-based agent:

```bash
python examples/run_gym_demo.py --agent rule_based
```

Available demo agents are:

- `random`
- `stop`
- `rule_based`
- `world_model`

The `KeyboardAgent` interface exists as a placeholder for a future interactive backend.

The demo loads `configs/scenarios/forest_trail_001.yaml`, runs an agent, and prints metrics such as `success`, `collision_count`, `path_length`, `total_reward`, `average_terrain_risk`, and `control_smoothness`.

## Gymnasium Wrapper

M12 adds a Gymnasium-compatible environment:

```python
from offroad_sim.rl import OffroadGymEnv

env = OffroadGymEnv(max_episode_steps=200)
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Smoke test it with:

```bash
python examples/run_gymnasium_env.py --steps 20
```

## Evaluation Metrics

`offroad_sim.evaluation.MetricsTracker` computes the common episode metrics used by the backend and demo:

- `success`
- `total_reward`
- `episode_length`
- `time_to_goal`
- `path_length`
- `average_speed`
- `max_speed`
- `collision_count`
- `rollover`
- `max_pitch`
- `max_roll`
- `average_terrain_risk`
- `control_smoothness`

`offroad_sim.evaluation.run_episode` is the shared runner used by examples, the CLI, and the dashboard API.

## World Models

M11 adds the world-model API and a deterministic baseline:

- `BaseWorldModel`
- `WorldModelPrediction`
- `SimpleKinematicWorldModel`
- `WorldModelAgent`

Run the world-model agent:

```bash
python examples/run_world_model_agent.py --max-steps 500
```

The baseline predicts a short bicycle-model rollout and samples terrain risk from the current observation. Learned models can replace it without changing the agent/backend interfaces.

## Episode Recording

Save a demo episode:

```bash
python examples/run_gym_demo.py --agent rule_based --record outputs/episodes/demo_001
```

Replay the saved episode:

```bash
python examples/replay_episode.py outputs/episodes/demo_001
```

The episode format is:

```text
episode_dir/
  metadata.json
  metrics.json
  steps.jsonl
  arrays/
```

Observation arrays are skipped by default to keep recordings small. Pass `--record-arrays` to persist arrays such as `local_bev` as `.npy` files.

## Dataset Replay

M8 adds a dataset adapter layer so replay does not depend on one fixed file layout.

The normalized path is:

```text
physical dataset layout -> DatasetAdapter -> DatasetSequence/DatasetFrame -> DatasetReplayBackend
```

Dynamic dataset switching works through `DatasetRegistry`:

- pass `adapter="offroad_sim_v1"` when the format is known;
- omit `adapter` to let the registry inspect `dataset.yaml`/`manifest.yaml` and auto-detect a loader;
- add future adapters for KITTI, rosbag exports, CSV/image folders, or custom logs without changing `DatasetReplayBackend`.

Create a tiny mock dataset:

```bash
python scripts/create_mock_dataset.py outputs/mock_dataset_m8 --frames 6
```

Replay it through the backend:

```bash
python examples/run_dataset_replay.py outputs/mock_dataset_m8 --load-assets
```

The built-in `offroad_sim_v1` mock layout is:

```text
dataset_root/
  dataset.yaml
  sequences/
    seq_0001/
      poses.csv
      metadata.json
      calibration.json
      images/
      depth/
      lidar/
      bev/
      terrain/
      labels/
```

## BeamNG Backend

`BeamNGBackend` is an optional adapter for BeamNG.tech through `beamngpy`. The project does not include BeamNG.tech or BeamNG assets, and importing `offroad_sim` does not require BeamNG to be installed.

Before using it for a real run:

```powershell
python -m pip install beamngpy
$env:BNG_HOME = "D:\programs\OffroadSimBench\BeamNG\BeamNG.tech.v0.38.3.0"
```

The backend exposes the expected simulator lifecycle:

- `connect()`
- `load_scenario()`
- `spawn_vehicle()`
- `attach_sensors()`
- `reset()`
- `step()`
- `get_observation()`
- `get_metrics()`
- `close()`

See `docs/beamng_backend.md` for the current integration boundary and the next sensor-mapping pass.

## UE5 Backend

`UE5Backend` is a TCP JSON bridge placeholder for a future Unreal runtime. It currently ships with a local `MockUE5Server` so the protocol can be tested without Unreal Engine:

```bash
python examples/run_mock_ue5_backend.py
```

See `docs/ue5_backend.md` for the command and observation JSON schema.

## Dashboard

M13/M14 add a local FastAPI API and React/Vite dashboard. Phase 2 extends it
into a live demo console backed by `/stream_episode` and recorded replay APIs.

Start the API:

```bash
uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```bash
cd dashboard/frontend
npm install
npm run dev
```

The default frontend expects the API at `http://127.0.0.1:8000`. Use `VITE_API_BASE` to point it elsewhere.

Phase 2 acceptance can be run with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase2_acceptance.ps1
```

See `docs/phase2_demo_plan.md` for the visible demo checklist.

## Next Stage

The next stage should focus on real simulator runtime validation: BeamNG scenario loading, sensor payload mapping, controller stepping, and a small end-to-end run once the local BeamNG runtime is ready.
