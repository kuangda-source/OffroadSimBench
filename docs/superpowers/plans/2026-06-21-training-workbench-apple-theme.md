# Training Workbench Apple Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Dataset and Training workbench so built-in and manifest training configs are easier to understand, recent loss curves are visible, and the desktop GUI moves toward an Apple-style light visual system.

**Architecture:** Keep runtime logic in `desktop_app/services.py` and GUI composition in `desktop_app/qt_main.py`. Add summary widgets around the existing training preset and training results flow rather than inventing a second training system. Replace the current dark stylesheet with a light, restrained set of colors and spacing.

**Tech Stack:** PySide6, pytest, existing desktop service catalog, existing `TrainingCurveWidget`.

---

### Task 1: Training Workbench Summary Tests

**Files:**
- Modify: `tests/test_desktop_visible_demo.py`
- Modify: `desktop_app/qt_main.py`

- [x] Write a failing GUI test that the Dataset and Training page exposes `training_preset_summary`, a "Training config summary" label, and a "Latest metric curve" label.
- [x] Write a failing test that selecting a manifest training preset updates the summary text with the preset description and manifest path.
- [x] Implement the summary widgets and preset-sync behavior.
- [x] Run the targeted GUI tests.

### Task 2: Apple-Style Theme Tests

**Files:**
- Modify: `tests/test_desktop_visible_demo.py`
- Modify: `desktop_app/qt_main.py`

- [x] Update stylesheet tests to require Apple-style light tokens: `#f5f5f7`, `#ffffff`, `#1d1d1f`, and `#007aff`.
- [x] Replace the dark stylesheet with light backgrounds, neutral cards, blue primary buttons, softer borders, and visible selected tabs.
- [x] Ensure text remains readable in training curves and preview panels.
- [x] Run the targeted style tests.

### Task 3: Verification

**Files:**
- Validate all modified files.

- [x] Run `python -m pytest -q`.
- [x] Run `python examples/run_gym_demo.py --agent rule_based --max-steps 1200`.
- [x] Run `python -m offroad_sim.cli list`.
- [x] Run the Qt offscreen window smoke.
- [ ] Commit and push this slice, excluding the pre-existing uncommitted task YAML.
