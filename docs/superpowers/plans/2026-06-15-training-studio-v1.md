# Training Studio V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the "Dataset & Training" GUI page into a training-focused workbench with presets, reproducible run records, and visible model artifacts, without coupling it to BeamNG driving.

**Architecture:** Keep data/training orchestration in `desktop_app/services.py`, and keep PySide6 UI composition in `desktop_app/qt_main.py`. Training actions write a `training_run.json` next to the produced artifact, and `catalog_snapshot()` exposes a `training_runs` list so the GUI can show recent runs.

**Tech Stack:** Python dataclasses, PySide6 widgets, existing dataset adapters, existing `tiny_learned` and LE-WM cost-model scripts, pytest.

---

### Task 1: Service Layer Training Presets And Run Records

**Files:**
- Modify: `desktop_app/services.py`
- Modify: `tests/test_desktop_services.py`

- [ ] **Step 1: Write failing service tests**

Add tests that assert:
- `services.training_preset_entries()` contains `tiny_world_model`, `lewm_cost_model`, `stablewm_hdf5`, and visible unfinished entries for `lewm_full_self_supervised`, `tdmpc2_adapter`, and `dreamerv3_adapter`.
- Training output helpers can write a `training_run.json` with dataset, adapter, sequence, preset, status, artifact path, and metrics.
- `services.training_run_entries(tmp_path)` discovers the saved run.

- [ ] **Step 2: Run service tests and verify red**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_services.py::test_training_preset_entries_include_available_and_future_models tests/test_desktop_services.py::test_training_run_record_is_discoverable -q`

Expected: FAIL because these service helpers do not exist yet.

- [ ] **Step 3: Implement minimal service helpers**

Add:
- `TRAINING_RUN_FILENAME = "training_run.json"`
- `training_preset_entries()`
- `write_training_run_record(...)`
- `training_run_entries(root=None)`

Extend `catalog_snapshot()` with `training_presets` and `training_runs`.

- [ ] **Step 4: Run service tests and verify green**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_services.py -q`

Expected: PASS.

### Task 2: Record Existing Training Actions

**Files:**
- Modify: `desktop_app/services.py`
- Modify: `tests/test_desktop_services.py`

- [ ] **Step 1: Write failing tests**

Patch the existing training helpers and assert:
- `train_tiny_world_model()` writes `training_run.json`.
- `train_lewm_cost_model()` writes `training_run.json`.
- `export_lewm_hdf5()` writes `training_run.json` beside the exported HDF5.

- [ ] **Step 2: Run targeted tests and verify red**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_services.py -q`

Expected: FAIL until the functions write run records.

- [ ] **Step 3: Implement recording**

Update the three training/export functions to call `write_training_run_record()` after success. Keep the command outputs intact and add a `training_run_path` field to returned payloads.

- [ ] **Step 4: Run service tests and verify green**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_services.py -q`

Expected: PASS.

### Task 3: GUI Training Studio Controls

**Files:**
- Modify: `desktop_app/qt_main.py`
- Modify: `tests/test_desktop_visible_demo.py`

- [ ] **Step 1: Write failing GUI tests**

Add tests that assert:
- The dataset/training page exposes a `Training preset` selector and one `Start training/export` primary button.
- The page exposes a training run list.
- Selecting the preset and starting it dispatches the correct existing action without BeamNG-driving fields.

- [ ] **Step 2: Run GUI tests and verify red**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_visible_demo.py -q`

Expected: FAIL until controls exist.

- [ ] **Step 3: Implement GUI controls**

Add:
- `self.training_preset_combo`
- `self.training_run_list`
- `self.run_training_preset()`
- `_fill_training_run_list()`
- `_load_selected_training_run()`

Keep existing explicit action buttons as secondary controls for now, but make preset + start the primary path.

- [ ] **Step 4: Run GUI tests and verify green**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_visible_demo.py -q`

Expected: PASS.

### Task 4: Verification And Push

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run required verification**

Run:
- `.\\.conda\\offroad-sim-bench\\python.exe -m pytest -q`
- `.\\.conda\\offroad-sim-bench\\python.exe examples\\run_gym_demo.py --agent rule_based --max-steps 1200`
- `.\\.conda\\offroad-sim-bench\\python.exe -m offroad_sim.cli list`
- `$env:QT_QPA_PLATFORM='offscreen'; .\\.conda\\offroad-sim-bench\\python.exe -c "from PySide6.QtWidgets import QApplication; from desktop_app.qt_main import MainWindow; app=QApplication([]); w=MainWindow(); print(w.windowTitle()); w.close()"`

Expected: all commands exit 0.

- [ ] **Step 2: Commit and push**

Stage only `desktop_app`, `tests`, and this plan. Do not stage `configs/tasks/beamng_johnson_valley_nav_001.yaml`.
