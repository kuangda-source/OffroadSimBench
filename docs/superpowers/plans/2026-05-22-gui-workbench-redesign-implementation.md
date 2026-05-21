# GUI Workbench Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the desktop GUI into a guided demo launcher plus Dataset/Training and BeamNG Simulation workbenches.

**Architecture:** Keep the existing PySide6 `MainWindow` shell and service layer, but reduce the top-level navigation to four pages. Reuse existing controls and worker-thread actions, moving them into clearer workbench groups instead of rewriting runtime logic.

**Tech Stack:** Python, PySide6, pytest, existing `desktop_app.services` APIs.

---

### Task 1: GUI Structure Tests

**Files:**
- Modify: `tests/test_desktop_visible_demo.py`

- [ ] **Step 1: Write failing tests**

Add tests that create `MainWindow` offscreen and assert:

```python
def test_gui_uses_guided_demo_and_two_workbenches() -> None:
    _ensure_app()
    window = MainWindow()
    nav_texts = [button.text() for button in window.nav_buttons]
    assert nav_texts == ["总览", "数据集与训练", "BeamNG 仿真", "实验记录"]
    window.close()

def test_gui_overview_is_guided_demo_launcher() -> None:
    _ensure_app()
    window = MainWindow()
    overview = window.page_stack.widget(0)
    labels = [label.text() for label in overview.findChildren(QLabel)]
    buttons = [button.text() for button in overview.findChildren(QPushButton)]
    assert "Demo preset" in labels
    assert "BeamNG region task" in labels
    assert "World model config" in labels
    assert "Run guided demo" in buttons
    assert "Open Dataset & Training" in buttons
    assert "Open BeamNG Simulation" in buttons
    assert "Backend" not in labels
    assert "Scenario" not in labels
    assert "Agent" not in labels
    window.close()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_desktop_visible_demo.py::test_gui_uses_guided_demo_and_two_workbenches tests/test_desktop_visible_demo.py::test_gui_overview_is_guided_demo_launcher -q
```

Expected: failures because the GUI still has six pages and the overview still exposes runtime internals.

### Task 2: Main Window Information Architecture

**Files:**
- Modify: `desktop_app/qt_main.py`

- [ ] **Step 1: Add missing top-level controls**

Add shared controls for demo preset and BeamNG workbench model config:

```python
self.demo_preset_combo = self._combo()
self.beamng_model_config_combo = self._combo()
self.beamng_task_combo = self._combo(editable=True)
```

- [ ] **Step 2: Replace top-level pages**

Set `PAGE_TITLES` to four entries:

```python
[
    ("总览", "通过引导式 demo 快速检查配置并运行可视演示。"),
    ("数据集与训练", "导入/预览数据集，训练算法模型，并检查训练或推理结果。"),
    ("BeamNG 仿真", "编辑地图任务，选择模型配置，运行自动驾驶并评估。"),
    ("实验记录", "浏览 episode、轨迹、指标和日志。"),
]
```

Build the stack with:

```python
self.page_stack.addWidget(self._build_overview_page())
self.page_stack.addWidget(self._build_dataset_training_page())
self.page_stack.addWidget(self._build_beamng_simulation_page())
self.page_stack.addWidget(self._build_records_page())
```

- [ ] **Step 3: Convert overview to guided demo launcher**

Keep only demo preset, task, model config, planner, environment status, run button, and workbench navigation buttons. Remove backend/scenario/agent fields from the overview.

- [ ] **Step 4: Combine dataset and training controls**

Create `_build_dataset_training_page()` that places dataset import/preview controls, model config/training controls, and model output in one workbench. Remove the one-click ORFD-to-BeamNG button from visible GUI controls.

- [ ] **Step 5: Combine BeamNG controls**

Create `_build_beamng_simulation_page()` with task selection, model config selection, region editor, run button, BeamNG check, terrain draft export, terrain preview, and BeamNG summary.

- [ ] **Step 6: Run tests and verify pass**

Run:

```powershell
python -m pytest tests/test_desktop_visible_demo.py -q
```

Expected: pass.

### Task 3: Cleanup Generated Artifacts

**Files:**
- Delete generated directories under `outputs/`, not committed source files.
- Preserve `outputs/region_navigation/johnson_valley_nav_test_train_v2_validated/model/lewm_cost_object.ckpt`.
- Preserve `outputs/region_navigation/johnson_valley_nav_test_train_v2_validated/model/metadata.json`.
- Preserve `configs/tasks/beamng_johnson_valley_nav_test.yaml`.

- [ ] **Step 1: List cleanup candidates**

Run:

```powershell
Get-ChildItem outputs -Directory
Get-ChildItem outputs\region_navigation -Directory
Get-ChildItem outputs\episodes -Directory
```

- [ ] **Step 2: Remove generated non-demo output directories**

Delete stale `outputs/episodes`, `outputs/gui_previews`, old `outputs/region_navigation/*` directories except `johnson_valley_nav_test_train_v2_validated`, and other generated model/checkpoint directories that are not the validated demo checkpoint.

- [ ] **Step 3: Verify default checkpoint remains**

Run:

```powershell
Test-Path outputs\region_navigation\johnson_valley_nav_test_train_v2_validated\model\lewm_cost_object.ckpt
```

Expected: `True`.

### Task 4: Documentation and Review

**Files:**
- Modify: `README.md`
- Modify: `docs/desktop_gui.md`

- [ ] **Step 1: Update docs**

Replace references to one-click ORFD-to-BeamNG workflow as a visible GUI button with the new guided demo/workbench structure.

- [ ] **Step 2: Full validation**

Run:

```powershell
python -m pytest -q
python examples/run_gym_demo.py --agent rule_based --max-steps 1200
python -m offroad_sim.cli list
$env:QT_QPA_PLATFORM='offscreen'; python -c "from PySide6.QtWidgets import QApplication; from desktop_app.qt_main import MainWindow; app=QApplication([]); w=MainWindow(); print(w.windowTitle()); print([b.text() for b in w.nav_buttons])"
```

- [ ] **Step 3: Review diff**

Run:

```powershell
git diff --check
git diff --stat
git status --short
```

Confirm only intended source/docs/test files are staged, and do not stage unrelated user changes in `configs/tasks/beamng_johnson_valley_nav_001.yaml`.
