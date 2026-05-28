# Region Self-Supervised World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a route-free BeamNG self-supervised world-model training and direct-control evaluation loop.

**Architecture:** Add two focused agents: `region_explorer` for collection and `world_model_direct` for route-free MPC. Add one service request that trains `tiny_learned` from recorded BeamNG transitions and evaluates without expert route metadata.

**Tech Stack:** Python, PySide6 service layer, pytest, existing `TinyLearnedWorldModel`, existing BeamNG backend.

---

### Task 1: Agent Boundaries

**Files:**
- Create: `offroad_sim/agents/region_explore.py`
- Create: `offroad_sim/agents/world_model_direct.py`
- Modify: `offroad_sim/agents/registry.py`
- Test: `tests/test_agents.py`

- [ ] Add tests for `region_explorer` temporary goals staying inside the region.
- [ ] Add tests for `world_model_direct` ignoring `Observation.info["route"]`.
- [ ] Register both agents in the default agent registry.

### Task 2: Self-Supervised Service

**Files:**
- Modify: `desktop_app/services.py`
- Test: `tests/test_region_self_supervised_world_model.py`

- [ ] Add `RegionSelfSupervisedWorldModelRequest`.
- [ ] Add trace-to-`DatasetSequence` conversion.
- [ ] Add route-free scenario helper.
- [ ] Add `run_region_self_supervised_world_model()`.

### Task 3: GUI Hook

**Files:**
- Modify: `desktop_app/qt_main.py`
- Test: `tests/test_desktop_visible_demo.py`

- [ ] Add a button under Dataset and Training for region self-supervised training.
- [ ] Add a BeamNG Simulation button for route-free direct world-model evaluation.
- [ ] Ensure requests use selected task, model path, and planner settings.

### Task 4: Verification

**Commands:**
- `python -m pytest -q`
- `python examples/run_gym_demo.py --agent rule_based --max-steps 1200`
- `python -m offroad_sim.cli list`
- `QT_QPA_PLATFORM=offscreen python -c "from PySide6.QtWidgets import QApplication; from desktop_app.qt_main import MainWindow; app=QApplication([]); w=MainWindow(); print(w.windowTitle())"`

