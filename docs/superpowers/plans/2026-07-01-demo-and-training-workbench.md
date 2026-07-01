# Standard Demo And Training Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seal one standard Johnson Valley BeamNG demo and turn Dataset and Training into a usable smoke-tested training workbench.

**Architecture:** Keep BeamNG-specific execution in `desktop_app.services` and GUI orchestration in `desktop_app.qt_main`. Add a small demo catalog, a reusable multi-run acceptance service, and a script wrapper. Training workbench changes stay behind the existing dataset/trainer/training-config services.

**Tech Stack:** Python, PySide6, pytest, existing OffroadSimBench services and YAML configs.

---

### Task 1: Standard Demo Catalog And Homepage

**Files:**
- Modify: `desktop_app/services.py`
- Modify: `desktop_app/qt_main.py`
- Test: `tests/test_desktop_services.py`
- Test: `tests/test_desktop_visible_demo.py`

- [x] Add a `demo_config_entries()` service with a default `johnson_valley_standard_demo` row pointing at `configs/tasks/beamng_johnson_valley_nav_001.yaml`, `DEFAULT_WORLD_MODEL_CONFIG_ID`, and `navigation_mpc`.
- [x] Update `DEFAULT_NAVIGATION_TASK_PATH` to `beamng_johnson_valley_nav_001.yaml`.
- [x] Add tests proving the default demo config resolves the official task path and appears first in catalogs.
- [x] Simplify the overview page so labels are only `Demo config`, plus a run button and result/status panels.
- [x] Add tests proving `BeamNG region task`, `World model config`, and `Planner` no longer appear on the overview page.

### Task 2: Demo Acceptance Service And Script

**Files:**
- Modify: `desktop_app/services.py`
- Create: `scripts/demo_acceptance.py`
- Test: `tests/test_desktop_services.py`

- [x] Add request/result dataclasses for 1-3 run demo acceptance.
- [x] Add a service that runs the selected demo 1-3 times through the existing region navigation loop and summarizes goal reached, collision count, final distance, trajectory length, average speed, and recovery usage.
- [x] Add tests with patched BeamNG execution and synthetic episode traces to prove metrics are aggregated correctly.
- [x] Add a CLI script that prints a compact JSON acceptance report and exits non-zero if any run fails the acceptance gate.

### Task 3: Training Workbench Smoke Defaults

**Files:**
- Modify: `desktop_app/services.py`
- Test: `tests/test_training_manifest_services.py`

- [x] Add a built-in `smoke_tiny_world_model` training config that uses a generated tiny ORFD-style dataset under `outputs/training_studio_smoke/datasets/mock_orfd`.
- [x] Ensure validation and execution create the smoke dataset when that config is selected.
- [x] Add tests proving the default config can run with the generic trainer manifest and writes `training_run.json` with metric history.

### Task 4: Training Workbench Usability

**Files:**
- Modify: `desktop_app/qt_main.py`
- Test: `tests/test_desktop_visible_demo.py`

- [x] Add a visible training log panel for the latest run using recorded stdout/stderr paths when available.
- [x] Add a button to register the latest successful model artifact as a world-model config when the artifact is compatible.
- [x] Add tests proving the result panel shows metrics, artifact path, logs, and world-model registration state.

### Task 5: Verification And Docs

**Files:**
- Modify: `docs/desktop_gui.md`
- Modify: `README.md`

- [x] Document the standard demo and the `scripts/demo_acceptance.py --runs 1` command.
- [x] Document the smoke training config and external trainer contract.
- [x] Run `python -m pytest -q`.
- [x] Run `python examples/run_gym_demo.py --agent rule_based --max-steps 1200`.
- [x] Run `python -m offroad_sim.cli list`.
- [x] Run the PySide6 offscreen smoke command.

### Verification Notes

- `python -m pytest -q`: 283 passed, 1 skipped.
- `python examples/run_gym_demo.py --agent rule_based --max-steps 1200`: success, 0 collisions.
- `python -m offroad_sim.cli list`: completed and listed BeamNG, dataset replay, gym, agents, world models, planners, and algorithms.
- PySide6 offscreen smoke printed `OffroadSimBench Desktop`.
- `python scripts\demo_acceptance.py --demo-config johnson_valley_standard_demo --runs 1 --max-steps 520 --beamng-gfx vk --planner-horizon 6 --planner-samples 32 --planner-iterations 3`: accepted, goal success true, collision count 0, final distance 11.2289 m, trajectory length 183.9354 m, average speed 7.2482 m/s, recovery false.
- `services.run_training_config_job("smoke_tiny_world_model")`: produced `outputs/training_studio_smoke/models/tiny_world_model/training_run.json` with history keys `frame_count`, `recorded_action_sample_count`, `sample_count`, `sequence_count`, `train_mse`, and `train_rmse`.
