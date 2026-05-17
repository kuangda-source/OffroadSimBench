# Pluggable Model Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first version of the capability-based algorithm adapter layer, with local LE-WM cost model as the reference implementation.

**Architecture:** Add `offroad_sim.algorithms` for manifests, adapter capability interfaces, registry discovery, and a built-in `local_lewm_cost` adapter. Keep runtime execution through existing `OffroadAgent`, `OffroadSimBackend`, BeamNG services, and LE-WM CEM planner instead of adding a parallel simulator path.

**Tech Stack:** Python dataclasses, existing YAML loader, existing StableWM HDF5 export and LE-WM cost training scripts, PySide6 desktop service catalog, argparse CLI.

---

### Task 1: Adapter Core And Manifest Parsing

**Files:**
- Create: `offroad_sim/algorithms/base.py`
- Create: `offroad_sim/algorithms/manifest.py`
- Create: `offroad_sim/algorithms/__init__.py`
- Test: `tests/test_algorithm_adapters.py`

- [x] **Step 1: Write failing tests**

Add tests that import `AlgorithmCapabilities`, `AlgorithmManifest`, and `AlgorithmAdapter`, parse a manifest dict, and verify unsupported capabilities raise `UnsupportedCapabilityError`.

- [x] **Step 2: Run tests and verify RED**

Run: `.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_algorithm_adapters.py -q`

Expected: import failure because `offroad_sim.algorithms` does not exist.

- [x] **Step 3: Implement core dataclasses**

Implement capability booleans, request/result dataclasses, `UnsupportedCapabilityError`, and manifest parsing with clear validation errors.

- [x] **Step 4: Run tests and verify GREEN**

Run the same pytest command and expect the new tests to pass.

### Task 2: Registry Discovery And Built-In LE-WM Adapter

**Files:**
- Create: `offroad_sim/algorithms/registry.py`
- Create: `offroad_sim/algorithms/builtins/__init__.py`
- Create: `offroad_sim/algorithms/builtins/local_lewm_cost.py`
- Modify: `offroad_sim/algorithms/__init__.py`
- Test: `tests/test_algorithm_adapters.py`

- [x] **Step 1: Write failing tests**

Add tests that `default_algorithm_registry()` exposes `local_lewm_cost`, discovers a temporary `algorithm.yaml + adapter.py` package, and can run the built-in adapter's data preparation on a tiny recorded episode.

- [x] **Step 2: Run tests and verify RED**

Run: `.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_algorithm_adapters.py -q`

Expected: registry or adapter symbols are missing.

- [x] **Step 3: Implement registry and built-in adapter**

Implement built-in registration, optional local folder discovery, dynamic entrypoint loading, and LE-WM data/train wrappers around existing scripts.

- [x] **Step 4: Run tests and verify GREEN**

Run the same pytest command and expect the tests to pass.

### Task 3: Service, GUI, And CLI Integration

**Files:**
- Modify: `desktop_app/services.py`
- Modify: `desktop_app/qt_main.py`
- Modify: `offroad_sim/cli.py`
- Test: `tests/test_desktop_services.py`
- Test: `tests/test_desktop_visible_demo.py`
- Test: `tests/test_algorithm_adapters.py`

- [x] **Step 1: Write failing tests**

Add tests that the desktop catalog exposes algorithms, the BeamNG LE-WM closed loop uses the selected algorithm adapter, the GUI contains an algorithm choice, and the CLI can list/inspect algorithms.

- [x] **Step 2: Run tests and verify RED**

Run: `.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_desktop_services.py tests\test_desktop_visible_demo.py tests\test_algorithm_adapters.py -q`

Expected: missing algorithms catalog, missing GUI control, or missing CLI command.

- [x] **Step 3: Implement integration**

Add algorithms to `catalog_snapshot()`, expose a GUI combo, route `BeamNGMapLeWMClosedLoopRequest` through the selected adapter, and add `offroad-sim algorithms list/inspect/prepare-data/train`.

- [x] **Step 4: Run tests and verify GREEN**

Run the same pytest command and expect the tests to pass.

### Task 4: Documentation And Acceptance

**Files:**
- Modify: `README.md`
- Modify: `docs/le_wm_integration.md`
- Modify: `docs/beamng_backend.md`

- [x] **Step 1: Update docs**

Document the `algorithm.yaml + adapter.py` contract, CLI commands, and the local LE-WM reference adapter.

- [x] **Step 2: Run full validation**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m compileall -q offroad_sim desktop_app scripts tests
.\.conda\offroad-sim-bench\python.exe -m pytest -q
.\.conda\offroad-sim-bench\python.exe examples\run_gym_demo.py --agent rule_based --max-steps 1200
.\.conda\offroad-sim-bench\python.exe -m offroad_sim.cli list
$env:QT_QPA_PLATFORM='offscreen'; .\.conda\offroad-sim-bench\python.exe -c "from PySide6.QtWidgets import QApplication; from desktop_app.qt_main import MainWindow; app=QApplication([]); w=MainWindow(); print(w.windowTitle())"
```

- [x] **Step 3: Run local algorithm acceptance**

Run CLI list/inspect for `local_lewm_cost`, then run a small BeamNG LE-WM closed-loop smoke if BeamNG is available.

- [x] **Step 4: Commit and push**

Commit with message `Add pluggable algorithm adapters` and push `main`.
