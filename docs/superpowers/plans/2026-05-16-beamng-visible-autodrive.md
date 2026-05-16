# BeamNG Visible Autodrive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a visible BeamNG autonomous-driving demo where a selected dataset and a selected world model/planner drive a real BeamNG vehicle through the shared OffroadSimBench runner.

**Architecture:** Keep dataset, model, planner, agent, and backend selection behind the existing registries. Use BeamNG's existing drivable maps for the first visible demo, while ORFD or another dataset provides model training, perception samples, and route/risk conditioning. Add BeamNG-specific visualization and runtime options inside `offroad_sim/backends/`, not in the GUI or runner.

**Tech Stack:** Python 3.11, BeamNG.tech/beamngpy, PySide6, NumPy, PyTorch/stable-worldmodel optional paths, pytest with fake BeamNG runtime tests.

---

## Scope

This phase promotes the current Phase 3 smoke loop into a visible, operator-friendly BeamNG demo:

```text
ORFD or other dataset -> model/checkpoint -> switchable planner/agent -> BeamNG visible vehicle -> recorded episode + GUI status
```

The first accepted demo uses a BeamNG stock level with configurable start, route, and goal. Full ORFD scene reconstruction as a packaged BeamNG level is tracked as an extension after the visible vehicle loop is reliable.

## File Structure

- Modify `offroad_sim/scenarios/config.py`: preserve generic scenario config while documenting BeamNG metadata fields.
- Create `configs/scenarios/beamng_visible_autodrive.yaml`: local visible BeamNG route demo config.
- Modify `offroad_sim/evaluation/runner.py`: allow a vehicle config path/object to be passed into backend construction.
- Modify `offroad_sim/backends/beamng_backend.py`: vehicle spawn pose, route markers, follow camera, sensor polling, collision/damage metrics, and visible debug options.
- Create `offroad_sim/agents/route_world_model.py`: route waypoint agent that still uses switchable world models/planners.
- Modify `offroad_sim/agents/registry.py`: register `route_world_model`.
- Create `scripts/run_beamng_visible_demo.py`: CLI entrypoint for local visual acceptance.
- Create `scripts/phase4_visible_beamng_acceptance.ps1`: repeatable local acceptance script.
- Modify `desktop_app/services.py`: expose visible BeamNG demo service and status payloads.
- Modify `desktop_app/qt_main.py`: add a focused BeamNG visible demo action and status panel.
- Create tests:
  - `tests/test_beamng_visible_config.py`
  - `tests/test_beamng_backend_visible.py`
  - `tests/test_route_world_model_agent.py`
  - `tests/test_desktop_visible_demo.py`

## Task 1: Visible BeamNG Scenario Contract

**Files:**
- Create: `configs/scenarios/beamng_visible_autodrive.yaml`
- Modify: `offroad_sim/scenarios/config.py`
- Test: `tests/test_beamng_visible_config.py`

- [ ] **Step 1: Write failing config tests**

Add tests that load `beamng_visible_autodrive.yaml` and assert:

```python
def test_beamng_visible_scenario_exposes_route_metadata():
    scenario = load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml")
    assert scenario.backend == "beamng"
    assert scenario.map == "west_coast_usa"
    assert scenario.metadata["beamng"]["level"] == "west_coast_usa"
    assert len(scenario.metadata["beamng"]["route"]) >= 3
    assert scenario.metadata["beamng"]["camera_mode"] == "orbit"
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_beamng_visible_config.py -q
```

Expected: fails because the scenario file does not exist.

- [ ] **Step 3: Add the scenario YAML**

Create a route using stock map coordinates:

```yaml
scenario_id: beamng_visible_autodrive
backend: beamng
map: west_coast_usa
weather: sunny
terrain:
  type: mixed_offroad
  difficulty: medium
task:
  max_time_sec: 180
  success_radius_m: 8.0
  start: [0.0, 0.0]
  goal: [80.0, 30.0]
metrics:
  collision: true
  rollover: true
  path_length: true
  terrain_risk: true
metadata:
  phase: 4
  demo_type: visible_beamng_autodrive
  beamng:
    level: west_coast_usa
    vehicle_model: pickup
    vehicle_start:
      pos: [0.0, 0.0, 0.5]
      rot_quat: [0.0, 0.0, 0.0, 1.0]
    route:
      - [0.0, 0.0]
      - [25.0, 5.0]
      - [55.0, 18.0]
      - [80.0, 30.0]
    camera_mode: orbit
    draw_route: true
    steps_per_action: 6
```

- [ ] **Step 4: Add small metadata helper functions**

Add helpers in `offroad_sim/scenarios/config.py`:

```python
def scenario_metadata_section(scenario: ScenarioConfig, name: str) -> dict[str, Any]:
    value = scenario.metadata.get(name, {})
    return dict(value) if isinstance(value, dict) else {}
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_beamng_visible_config.py -q
```

Commit:

```powershell
git add configs/scenarios/beamng_visible_autodrive.yaml offroad_sim/scenarios/config.py tests/test_beamng_visible_config.py
git commit -m "Add visible BeamNG scenario contract"
```

## Task 2: BeamNG Backend Visible Runtime

**Files:**
- Modify: `offroad_sim/backends/beamng_backend.py`
- Modify: `offroad_sim/evaluation/runner.py`
- Test: `tests/test_beamng_backend_visible.py`

- [ ] **Step 1: Write fake-beamng tests**

Test these behaviors without launching BeamNG:

```python
def test_beamng_backend_uses_vehicle_config_and_scenario_start(fake_beamngpy):
    backend = BeamNGBackend(connection=BeamNGConnectionConfig(launch=False), vehicle_config=load_vehicle_config("configs/vehicles/ugv_medium.yaml"))
    obs = backend.reset(load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml"))
    assert obs.info["backend"] == "beamng"
    assert fake_beamngpy.spawned_vehicle_model == "medium_offroad"
    assert fake_beamngpy.spawned_pos == (0.0, 0.0, 0.5)
```

```python
def test_beamng_backend_reports_motion_and_damage(fake_beamngpy):
    result = backend.step(Action(throttle=0.4))
    metrics = backend.get_metrics()
    assert metrics["episode_length"] == 1
    assert "distance_traveled" in metrics
    assert "collision_count" in metrics
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_beamng_backend_visible.py -q
```

Expected: failures around spawn pose, metrics, and fake APIs.

- [ ] **Step 3: Implement BeamNG metadata parsing**

Inside `BeamNGBackend`, read `scenario.metadata["beamng"]` for:

```python
{
    "level": "west_coast_usa",
    "vehicle_model": "pickup",
    "vehicle_start": {"pos": [0, 0, 0.5], "rot_quat": [0, 0, 0, 1]},
    "route": [[0, 0], [25, 5], [55, 18], [80, 30]],
    "draw_route": True,
    "camera_mode": "orbit",
    "steps_per_action": 6,
}
```

- [ ] **Step 4: Implement visible BeamNG setup**

Add best-effort calls guarded by `hasattr`:

```python
scenario.add_vehicle(vehicle, pos=pos, rot_quat=rot_quat)
bng.camera.set_free(pos=(x - 8.0, y - 8.0, z + 5.0), dir=(1.0, 1.0, -0.4))
bng.debug.add_spheres(route_points, radii=[0.8] * len(route_points), colors=[(0, 1, 0, 0.8)] * len(route_points))
```

Keep these calls non-fatal when a beamngpy version lacks a specific API.

- [ ] **Step 5: Pass vehicle config from runner**

Extend `run_episode()` with:

```python
vehicle: VehicleConfig | str | Path | None = None
```

Load it with `load_vehicle_config()` when a path is provided, then pass `vehicle_config=...` into BeamNG backend creation only when the backend accepts it.

- [ ] **Step 6: Run backend tests and commit**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_beamng_backend_visible.py tests\test_beamng_backend_import.py -q
```

Commit:

```powershell
git add offroad_sim/backends/beamng_backend.py offroad_sim/evaluation/runner.py tests/test_beamng_backend_visible.py
git commit -m "Add visible BeamNG runtime controls"
```

## Task 3: Route World-Model Agent

**Files:**
- Create: `offroad_sim/agents/route_world_model.py`
- Modify: `offroad_sim/agents/registry.py`
- Test: `tests/test_route_world_model_agent.py`

- [ ] **Step 1: Write route-following tests**

Add tests:

```python
def test_route_world_model_agent_advances_waypoints():
    agent = RouteWorldModelAgent(route=[(0, 0), (5, 0), (10, 0)], world_model_name="simple_kinematic")
    agent.reset({"route": [(0, 0), (5, 0), (10, 0)]})
    action = agent.act(observation_at(x=0.0, y=0.0, goal=(10.0, 0.0)))
    assert action.throttle > 0.0
    assert agent.diagnostics()["target_waypoint_index"] >= 1
```

```python
def test_route_world_model_agent_uses_planner_when_configured():
    agent = RouteWorldModelAgent(world_model_name="simple_kinematic", planner_name="world_model_cem", planner_config={"horizon": 3, "num_samples": 8, "iterations": 1})
    action = agent.act(observation_at(x=0.0, y=0.0, goal=(10.0, 0.0)))
    assert -1.0 <= action.steer <= 1.0
    assert agent.diagnostics()["planner"] == "world_model_cem"
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_route_world_model_agent.py -q
```

- [ ] **Step 3: Implement the agent**

Implement `RouteWorldModelAgent` as a thin wrapper over `WorldModelAgent`:

```python
class RouteWorldModelAgent(OffroadAgent):
    def __init__(self, route: list[tuple[float, float]] | None = None, waypoint_radius_m: float = 6.0, **world_model_kwargs: Any) -> None:
        self.route = route or []
        self.waypoint_radius_m = float(waypoint_radius_m)
        self.cursor = 0
        self.inner = WorldModelAgent(**world_model_kwargs)
```

Before delegating to `inner.act()`, replace `obs.goal` with the current route waypoint. Advance the cursor when the vehicle is within `waypoint_radius_m`.

- [ ] **Step 4: Register agent**

Add `route_world_model` to `default_agent_registry()` with factory `RouteWorldModelAgent`.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_route_world_model_agent.py tests\test_agents.py -q
```

Commit:

```powershell
git add offroad_sim/agents/route_world_model.py offroad_sim/agents/registry.py tests/test_route_world_model_agent.py
git commit -m "Add route world-model agent"
```

## Task 4: Dataset-to-BeamNG Demo Bundle

**Files:**
- Modify: `desktop_app/services.py`
- Create: `scripts/run_beamng_visible_demo.py`
- Test: `tests/test_desktop_visible_demo.py`

- [ ] **Step 1: Write service tests**

Patch service functions and assert the visible demo request keeps choices switchable:

```python
def test_visible_demo_request_keeps_dataset_model_and_backend_switchable():
    request = VisibleBeamNGDemoRequest(
        dataset_root="datasets/ORFD_Dataset_ICRA2022_ZIP",
        adapter="orfd",
        sequence_id="training/c2021_0228_1819",
        world_model_type="le_wm",
        world_model_path="outputs/models/lewm_orfd",
        planner="le_wm_cem",
        scenario="beamng_visible_autodrive",
        vehicle="configs/vehicles/ugv_medium.yaml",
    )
    payload = services.build_visible_beamng_demo_request(request)
    assert payload.agent == "route_world_model"
    assert payload.backend == "beamng"
    assert payload.world_model_type == "le_wm"
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_desktop_visible_demo.py -q
```

- [ ] **Step 3: Implement request dataclass and service**

Add:

```python
@dataclass(slots=True)
class VisibleBeamNGDemoRequest:
    dataset_root: str = ""
    adapter: str = "orfd"
    sequence_id: str = ""
    world_model_type: str = "le_wm"
    world_model_path: str = ""
    planner: str = "le_wm_cem"
    scenario: str = "beamng_visible_autodrive"
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    max_steps: int = 240
    seed: int = 7
```

The service calls `run_episode_from_request()` with backend `beamng`, agent `route_world_model`, the selected model/planner, and the visible scenario.

- [ ] **Step 4: Implement CLI script**

`scripts/run_beamng_visible_demo.py` should accept:

```powershell
--dataset-root datasets\ORFD_Dataset_ICRA2022_ZIP
--adapter orfd
--sequence-id training/c2021_0228_1819
--world-model-type le_wm
--world-model outputs\models\lewm_orfd
--planner le_wm_cem
--scenario beamng_visible_autodrive
--vehicle configs\vehicles\ugv_medium.yaml
--max-steps 240
```

It prints JSON with `episode_path`, `distance_traveled`, `steps`, `backend.connected`, and `success`.

- [ ] **Step 5: Run service tests and commit**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_desktop_visible_demo.py tests\test_desktop_services.py -q
```

Commit:

```powershell
git add desktop_app/services.py scripts/run_beamng_visible_demo.py tests/test_desktop_visible_demo.py
git commit -m "Add visible BeamNG demo service"
```

## Task 5: Desktop GUI Visible Demo Controls

**Files:**
- Modify: `desktop_app/qt_main.py`
- Test: `tests/test_desktop_layout.py`
- Test: `tests/test_desktop_visible_demo.py`

- [ ] **Step 1: Write GUI smoke assertions**

Extend GUI tests:

```python
def test_beamng_page_has_visible_demo_action():
    window = MainWindow()
    texts = [button.text() for button in window.findChildren(QPushButton)]
    assert "启动 BeamNG 可视自动驾驶" in texts
```

- [ ] **Step 2: Confirm test fails**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_desktop_layout.py -q
```

- [ ] **Step 3: Add GUI action**

Place the action in the BeamNG page, not the homepage:

```python
self._action_button("启动 BeamNG 可视自动驾驶", self.run_visible_beamng_demo, primary=True)
```

Show live payload in `beamng_summary` and update metric cards when the run finishes.

- [ ] **Step 4: Run GUI tests and commit**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.conda\offroad-sim-bench\python.exe -m pytest tests\test_desktop_layout.py tests\test_desktop_visible_demo.py -q
Remove-Item Env:\QT_QPA_PLATFORM
```

Commit:

```powershell
git add desktop_app/qt_main.py tests/test_desktop_layout.py tests/test_desktop_visible_demo.py
git commit -m "Expose visible BeamNG demo in desktop GUI"
```

## Task 6: Local Acceptance Script and Documentation

**Files:**
- Create: `scripts/phase4_visible_beamng_acceptance.ps1`
- Modify: `README.md`
- Modify: `docs/beamng_backend.md`
- Create: `docs/phase4_development_plan.md`
- Create: `docs/phase4_acceptance_report.md`

- [ ] **Step 1: Add acceptance script**

The script runs:

```powershell
python -m pytest -q
python -m offroad_sim.cli list --kind all
python scripts\run_beamng_visible_demo.py --dataset-root $OrfdRoot --adapter orfd --sequence-id $SequenceId --world-model-type $WorldModelType --world-model $WorldModelPath --planner $Planner --scenario beamng_visible_autodrive --vehicle configs\vehicles\ugv_medium.yaml --max-steps $MaxSteps
```

When `-BeamNGVisible` is omitted, it runs fake/mock tests only. When `-BeamNGVisible` is present, it requires BeamNG to launch and asserts:

```text
connected=true
steps >= 60
distance_traveled >= 10.0
episode_path exists
```

- [ ] **Step 2: Update docs**

Document two accepted commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1 -BeamNGVisible -OrfdRoot datasets\ORFD_Dataset_ICRA2022_ZIP -SequenceId training/c2021_0228_1819
```

- [ ] **Step 3: Run final verification**

Run:

```powershell
.\.conda\offroad-sim-bench\python.exe -m pytest -q
.\.conda\offroad-sim-bench\python.exe examples\run_gym_demo.py --agent rule_based --max-steps 1200
.\.conda\offroad-sim-bench\python.exe -m offroad_sim.cli list
$env:QT_QPA_PLATFORM='offscreen'
.\.conda\offroad-sim-bench\python.exe -c "from PySide6.QtWidgets import QApplication; from desktop_app.qt_main import MainWindow; app=QApplication([]); w=MainWindow(); print(w.windowTitle())"
Remove-Item Env:\QT_QPA_PLATFORM
```

- [ ] **Step 4: Run real visible acceptance on the local BeamNG machine**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase4_visible_beamng_acceptance.ps1 -BeamNGVisible -OrfdRoot datasets\ORFD_Dataset_ICRA2022_ZIP -SequenceId training/c2021_0228_1819 -WorldModelType le_wm -WorldModelPath outputs\models\lewm_orfd_real_c2021_0228_1819 -Planner le_wm_cem -MaxSteps 240
```

Expected: BeamNG launches a visible vehicle, the vehicle moves under autonomous control for at least 60 steps, and an episode is recorded.

- [ ] **Step 5: Commit and push**

Commit:

```powershell
git add scripts/phase4_visible_beamng_acceptance.ps1 README.md docs/beamng_backend.md docs/phase4_development_plan.md docs/phase4_acceptance_report.md
git commit -m "Document phase four visible BeamNG acceptance"
git push origin main
```

## Acceptance Criteria

- The GUI has a BeamNG page button named `启动 BeamNG 可视自动驾驶`.
- The CLI script can launch a visible BeamNG run using `beamng_visible_autodrive`.
- The agent/model/planner remain switchable through registries and request objects.
- The real BeamNG acceptance run records an episode with `connected=true`, `steps >= 60`, and `distance_traveled >= 10.0`.
- The codebase still works without BeamNG installed; fake BeamNG tests and import tests pass.
- The README clearly distinguishes the visible stock-map demo from full ORFD scene reconstruction.

## Risks and Mitigations

- ORFD does not provide a complete drivable BeamNG world. Mitigation: use ORFD for model/risk conditioning first, and BeamNG stock maps for visible closed-loop driving.
- ORFD often lacks true ego action logs. Mitigation: keep synthetic pose/action assumptions explicit in metadata and support other datasets with real odometry later.
- beamngpy APIs vary by version. Mitigation: isolate version-specific calls in `BeamNGBackend`, cover them with fake runtime tests, and keep missing visual helpers non-fatal.
- Full upstream LE-WM inference glue is separate from the current cost-model checkpoint. Mitigation: keep `le_wm_cem` accepted for Phase 4 and leave the upstream visual latent path behind the existing `LeWMWorldModel` adapter boundary.
