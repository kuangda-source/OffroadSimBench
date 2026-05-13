# AGENTS.md

## Project Name

OffroadSimBench

## Project Goal

Build a local off-road autonomous driving simulation and world-model evaluation platform.

The platform should support:
- multiple simulator backends;
- unified Agent API;
- unified Backend API;
- vehicle configuration import;
- scenario configuration import;
- dataset replay;
- Gym/heightmap training backend;
- BeamNG adapter;
- future UE5 adapter;
- episode recording;
- evaluation metrics;
- local dashboard.

## Development Principles

1. Do not hard-code logic to a single simulator.
2. Always keep `OffroadAgent` and `OffroadSimBackend` as the core abstractions.
3. Prefer small, testable modules.
4. Prefer typed Python with dataclasses or Pydantic models.
5. Do not add heavy dependencies unless necessary.
6. BeamNG and UE5 should be optional backends.
7. The first runnable version should work without BeamNG or UE5.
8. Use mock data where needed, but keep interfaces ready for real simulators.
9. Every major module should have basic tests.
10. Update README after each milestone.

## Validation Commands

After Python changes, run:

```bash
pytest
python examples/run_gym_demo.py
```

After frontend changes, run:

```bash
npm install
npm run dev
npm run build
```

## Expected Repository Structure

```text
offroad-sim-bench/
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
  README.md
  AGENTS.md
```

## Coding Style

- Python: use type hints.
- Configs: use YAML or JSON.
- Avoid hidden global state.
- Keep simulator-specific code inside `offroad_sim/backends/`.
- Keep agent-specific code inside `offroad_sim/agents/`.
- Keep metrics inside `offroad_sim/evaluation/`.
- Keep optional simulator integrations import-safe when their external tools are not installed.

