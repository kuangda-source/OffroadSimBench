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
ready through the shared backend registry and the dashboard `/beamng/status`
endpoint.

## Next BeamNG Pass

The next concrete BeamNG work should map `VehicleConfig.sensors` into beamngpy
sensors:

- camera -> RGB/depth image paths or arrays;
- lidar -> point cloud;
- imu -> acceleration/rotation;
- gps -> global pose;
- vehicle electrics -> speed and collision state.

The application layer should create this backend through `make_backend("beamng")`
or `default_backend_registry()`, not by importing BeamNG-specific code directly.
