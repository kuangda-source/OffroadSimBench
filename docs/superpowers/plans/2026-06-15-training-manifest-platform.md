# Training Manifest Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Dataset & Training workbench start training from dataset manifests and trainer manifests instead of hard-coded model-specific buttons.

**Architecture:** Add a `manifest_dataset` adapter that maps a YAML-described directory into normalized `DatasetSequence` objects. Add desktop-side trainer manifest parsing and a generic training job runner that records `training_run.json`, logs, parameters, metrics, and history. Keep built-in ORFD and existing LE-WM flows working while exposing manifest trainers in the GUI catalog.

**Tech Stack:** Python dataclasses, YAML configs, PySide6, subprocess JSON protocol, pytest.

---

### Task 1: Manifest Dataset Adapter

**Files:**
- Create: `offroad_sim/datasets/adapters/manifest.py`
- Modify: `offroad_sim/datasets/adapters/__init__.py`
- Modify: `offroad_sim/datasets/registry.py`
- Test: `tests/test_manifest_dataset_adapter.py`

- [x] Write failing tests for loading a dataset with `dataset_manifest.yaml`, listing a sequence, resolving image/depth/label globs, and preserving metadata.
- [x] Implement `ManifestDatasetAdapter`.
- [x] Register it in `default_dataset_registry()`.
- [x] Run targeted dataset adapter tests.

### Task 2: Trainer Manifest And Generic Runner

**Files:**
- Modify: `desktop_app/services.py`
- Test: `tests/test_training_manifest_services.py`

- [x] Write failing tests for parsing a `trainer.yaml`, exposing it as a training preset, generating default parameters, running a JSON-emitting command, and writing `training_run.json`.
- [x] Implement `trainer_manifest_entries()`, `load_trainer_manifest()`, `run_trainer_manifest_job(...)`, and command argument rendering.
- [x] Preserve existing training presets while appending manifest-based trainers.
- [x] Run targeted service tests.

### Task 3: GUI Wiring

**Files:**
- Modify: `desktop_app/qt_main.py`
- Test: `tests/test_desktop_visible_demo.py`

- [x] Write failing tests that selecting a manifest trainer exposes parameters and dispatches `run_trainer_manifest_job`.
- [x] Add trainer manifest combo population via existing `training_preset_combo`.
- [x] Add a compact parameter editor based on manifest parameter defaults.
- [x] Keep unavailable future presets read-only.
- [x] Run targeted GUI tests.

### Task 4: Docs And Verification

**Files:**
- Modify: `README.md`
- Verify all changed files.

- [x] Document `dataset_manifest.yaml` and `trainer.yaml` examples.
- [ ] Run full project validation commands from `AGENTS.md`.
- [ ] Commit and push only this feature's files; do not stage existing local task YAML edits.
