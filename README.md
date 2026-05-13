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

## Next Milestone

The next implementation step is M4: implement the lightweight Gym/Heightmap backend:

- `offroad_sim/backends/gym_heightmap_backend.py`
- `examples/run_gym_demo.py`
- `tests/test_gym_heightmap_backend.py`
