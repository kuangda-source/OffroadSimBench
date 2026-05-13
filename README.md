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
  backends/
    base.py
  core/
    types.py
  datasets/
  evaluation/
  replay/
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
scripts/
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

M4 adds a lightweight `GymHeightmapBackend` for fast local experiments without BeamNG or UE5. It generates a deterministic 2.5D terrain map with:

- heightmap;
- occupancy map;
- traversability map;
- terrain risk map.

Run a demo episode:

```bash
python examples/run_gym_demo.py --agent rule_based
```

The demo loads `configs/scenarios/forest_trail_001.yaml`, runs an agent, and prints metrics such as `success`, `collision_count`, `path_length`, and `total_reward`.

## Next Milestone

The next implementation step is M5: expand the basic agent set:

- `StopAgent`
- `KeyboardAgent` placeholder
- CLI-selectable agent behavior in demos
