# OffroadSimBench

OffroadSimBench is a local off-road autonomous driving simulation and world-model evaluation platform.

The first milestone focuses on a runnable Python foundation:

- a modular Python package named `offroad_sim`;
- a unified place for simulator backends;
- a unified place for driving agents;
- vehicle and scenario configuration directories;
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
  backends/
  core/
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
  vehicles/
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

## Next Milestone

The next implementation step is M2: define the core data types and abstract interfaces:

- `offroad_sim/core/types.py`
- `offroad_sim/agents/base.py`
- `offroad_sim/backends/base.py`
- `tests/test_core_interfaces.py`
