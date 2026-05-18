# BeamNG Backend

`BeamNGBackend` is an optional backend. OffroadSimBench does not ship BeamNG.tech,
BeamNG content, or a BeamNG license. The Python package must also remain importable
on machines that do not have BeamNG installed.

## Runtime Requirements

Install and configure these locally before using `BeamNGBackend.connect()` or
`BeamNGBackend.reset()`:

```powershell
python -m pip install beamngpy
$env:BNG_HOME = "D:\programs\OffroadSimBench\BeamNG\BeamNG.tech.v0.38.3.0"
```

`BNG_HOME` should point to the BeamNG.tech install directory that contains either
`BeamNG.tech.exe` or `Bin64/BeamNG.tech.x64.exe`.

On the local project workstation, the adapter also auto-detects
`BeamNG/BeamNG.tech*` under the repository root. You can inspect the resolved
runtime without launching BeamNG:

```powershell
python examples\check_beamng_runtime.py
```

For an explicit real connection smoke test:

```powershell
python examples\check_beamng_runtime.py --connect --steps 1
```

## Current Integration Boundary

The backend already exposes the shared simulator methods:

- `connect()`
- `load_scenario()`
- `spawn_vehicle()`
- `attach_sensors()`
- `reset()`
- `step()`
- `get_observation()`
- `get_metrics()`
- `close()`

Without `beamngpy` or `BNG_HOME`, the backend reports an unavailable status and
raises a clear `BeamNGUnavailableError` only when a real connection is attempted.
With both `beamngpy` and a detected executable present, the backend status becomes
ready through the shared backend registry and desktop GUI status panel
endpoint.

The Phase 4 visible demo uses `configs/scenarios/beamng_visible_autodrive.yaml`
on the stock `gridmap_v2` level. The scenario metadata defines the vehicle
model, spawn pose, route waypoints, route debug markers, camera setup, and
action step cadence. The normal run path is:

```powershell
python scripts\run_beamng_visible_demo.py --dataset-root datasets\ORFD_Dataset_ICRA2022_ZIP --adapter orfd --sequence-id training/c2021_0228_1819 --world-model-type le_wm --world-model outputs\models\lewm_orfd_real_c2021_0228_1819 --planner le_wm_cem --scenario beamng_visible_autodrive --vehicle configs\vehicles\ugv_medium.yaml --max-steps 600
```

For a local BeamNG-map LE-WM smoke loop, use:

```powershell
python scripts\run_beamng_lewm_closed_loop.py --collect-steps 160 --eval-steps 120 --output-dir outputs\beamng_map_lewm\demo
```

The loop collects an expert route episode on `gridmap_v2`, exports the
recording to stable-worldmodel HDF5 with state-derived actions, trains a local
LE-WM-compatible cost checkpoint, then launches a second visible BeamNG run
using `world_model_type=le_wm` and `planner=le_wm_cem`.

Internally this loop now uses the `local_lewm_cost` algorithm adapter, so future
models can replace the data-preparation/training/inference pieces through the
same pluggable adapter contract instead of changing BeamNG backend code.

For an explicit region/start/goal task, use:

```powershell
python scripts\run_region_navigation_loop.py --task configs\tasks\beamng_region_nav_001.yaml --algorithm local_lewm_cost --collect-steps 160 --eval-steps 120 --output-dir outputs\region_navigation\beamng_region_nav_001
```

The task YAML contains a polygonal region, start pose, goal radius, and expert
route. Collection uses the expert route for demonstration data; evaluation
omits the route from BeamNG metadata, leaving the backend to derive a direct
start-to-goal target from the task. Set `beamng.evaluation_drive_mode: manual`
to let the OffroadAgent/model commands control the vehicle during evaluation.
Set it to `ai_line` only for BeamNG-native visual smoke tests, because `ai_line`
uses BeamNG's own route follower. The BeamNG backend terminates the episode
when the vehicle enters the configured goal radius and records distance-to-goal
metrics.

The desktop GUI exposes a `编辑区域/起终点` dialog on the BeamNG page. It lets a
user click a 2D coordinate canvas to select the region polygon, start point,
goal point, and optional expert route waypoints, then saves a
`navigation_region_v1` YAML file under `configs/tasks/`.

The demo runs through the shared `route_world_model` agent so the selected
world model and planner remain replaceable without BeamNG-specific application
logic. For the visible smoke demo, BeamNG execution uses `drive_mode=ai_line`
on the configured route because this is the stable BeamNG-native way to show
the vehicle driving on the stock map; the agent and planner diagnostics are
still recorded for the same run.

For manual viewing, the visible script intentionally differs from acceptance
scripts: it waits briefly after loading BeamNG, adds a small wall-clock delay
between control steps, and does not close BeamNG by default. Add
`--close-beamng` for automated runs that should clean up the simulator process.
The default graphics backend is Vulkan (`gfx="vk"`) because the local
BeamNG.tech 0.38.3 Direct3D11 auto-launch path can produce a black render
window while simulation continues in the background. Use `--beamng-gfx dx11`
only when Direct3D11 is known to render correctly on the machine.

## Remaining BeamNG Work

Sensor attachment is best-effort across beamngpy versions. The next concrete
BeamNG work should harden `VehicleConfig.sensors` mapping into beamngpy sensors:

- camera -> RGB/depth image paths or arrays;
- lidar -> point cloud;
- imu -> acceleration/rotation;
- gps -> global pose;
- vehicle electrics -> speed and collision state.

Full ORFD scene reconstruction as a BeamNG level is also still a map-building
task. The current accepted visual loop uses real ORFD data for model selection
and a stock BeamNG level for the closed-loop vehicle demonstration.

The application layer should create this backend through `make_backend("beamng")`
or `default_backend_registry()`, not by importing BeamNG-specific code directly.
