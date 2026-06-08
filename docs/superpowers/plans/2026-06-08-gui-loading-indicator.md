# GUI Loading Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show clear animated feedback whenever a PySide6 GUI background task is running.

**Architecture:** Reuse `MainWindow._run_task()` as the single async task boundary. Add one global busy bar and label in the header so every existing background task automatically shows a loading state and hides it when the worker finishes or fails.

**Tech Stack:** Python, PySide6 `QProgressBar`, existing `TaskWorker`/`QThread`, pytest GUI smoke tests.

---

### Task 1: Test Busy State Lifecycle

**Files:**
- Modify: `tests/test_desktop_visible_demo.py`
- Modify: `desktop_app/qt_main.py`

- [ ] **Step 1: Write the failing test**

Add a test that calls `MainWindow._set_busy(True, "Start test")`, verifies the busy label and animated progress bar are visible, then calls `_set_busy(False)` and verifies both hide again.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_visible_demo.py::test_gui_busy_indicator_lifecycle -q`

Expected: FAIL because `MainWindow` does not yet expose the busy indicator widgets or `_set_busy()`.

- [ ] **Step 3: Implement minimal UI**

In `desktop_app/qt_main.py`, import `QProgressBar`, create `self.busy_label` and `self.busy_bar` in `_build_main_area()`, and add `_set_busy(active, label="")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_visible_demo.py::test_gui_busy_indicator_lifecycle -q`

Expected: PASS.

### Task 2: Wire Busy State Through `_run_task`

**Files:**
- Modify: `desktop_app/qt_main.py`
- Modify: `tests/test_desktop_visible_demo.py`

- [ ] **Step 1: Write the failing test**

Add a test that calls `_run_task(..., task_label="Start test")`, verifies the busy state appears immediately, releases the worker, and verifies the busy state hides after thread cleanup.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_visible_demo.py::test_gui_run_task_shows_and_hides_busy_indicator -q`

Expected: FAIL because `_run_task()` does not accept `task_label` and does not call `_set_busy()`.

- [ ] **Step 3: Implement task wiring**

Update `_run_task()` to accept `task_label: str = ""`, call `_set_busy(True, task_label or cleaned failure label)` before starting the thread, and connect both `worker.finished` and `worker.failed` to `_set_busy(False)`.

- [ ] **Step 4: Run targeted tests**

Run: `.\\.conda\\offroad-sim-bench\\python.exe -m pytest tests/test_desktop_visible_demo.py -q`

Expected: PASS.

### Task 3: Verification And Commit

**Files:**
- Verify: all Python and GUI tests required by `AGENTS.md`

- [ ] **Step 1: Run full verification**

Run:
- `.\\.conda\\offroad-sim-bench\\python.exe -m pytest -q`
- `.\\.conda\\offroad-sim-bench\\python.exe examples\\run_gym_demo.py --agent rule_based --max-steps 1200`
- `.\\.conda\\offroad-sim-bench\\python.exe -m offroad_sim.cli list`
- `$env:QT_QPA_PLATFORM='offscreen'; .\\.conda\\offroad-sim-bench\\python.exe -c "from PySide6.QtWidgets import QApplication; from desktop_app.qt_main import MainWindow; app=QApplication([]); w=MainWindow(); print(w.windowTitle()); w.close()"`

Expected: all commands exit 0.

- [ ] **Step 2: Commit and push**

Stage only the GUI loading files and this plan. Do not stage `configs/tasks/beamng_johnson_valley_nav_001.yaml`.
