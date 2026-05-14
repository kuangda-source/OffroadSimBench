# OffroadSimBench

OffroadSimBench is a local off-road autonomous driving simulation and world-model evaluation platform.

The current milestone focuses on a runnable Python foundation:

- a modular Python package named `offroad_sim`;
- shared core types for actions, observations, vehicle state, and step results;
- base interfaces for simulator backends and driving agents;
- vehicle and scenario configuration loading from YAML;
- future support for Gym/heightmap simulation, dataset replay, BeamNG, UE5, evaluation, replay, world models, RL, and a local dashboard.

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
    dataset_replay_backend.py
    gym_heightmap_backend.py
  core/
    types.py
  datasets/
    adapters/
    mock.py
    registry.py
    types.py
  evaluation/
    metrics.py
  replay/
    episode.py
  rl/
  scenarios/
  utils/
  vehicles/
  world_models/
configs/
  scenarios/
    forest_trail_001.yaml
  vehicles/
    ugv_medium.yaml
dashboard/
  backend/
  frontend/
docs/
examples/
  run_gym_demo.py
  run_dataset_replay.py
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

The `KeyboardAgent` interface exists as a placeholder for a future interactive backend.

The demo loads `configs/scenarios/forest_trail_001.yaml`, runs an agent, and prints metrics such as `success`, `collision_count`, `path_length`, `total_reward`, `average_terrain_risk`, and `control_smoothness`.

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

## Next Milestone

The next implementation step is M9: connect replayed datasets to richer world-model or evaluation workflows.
