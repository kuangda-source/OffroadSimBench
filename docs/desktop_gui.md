# Desktop GUI

The desktop GUI lives in `desktop_app/` and uses PySide6. It calls the same
local Python services as the CLI and acceptance scripts.

## Run

```powershell
python -m desktop_app.main
# or, after editable install refresh:
offroad-sim-gui
```

## Connected Features

- Runtime catalogs for backends, agents, world models, and planners.
- Dataset inspection through the registered dataset adapters.
- Episode execution through `offroad_sim.evaluation.run_episode`.
- Tiny learned world-model training through `TinyLearnedWorldModel.fit`.
- Episode list loading from `outputs/episodes`.
- BEV-style trajectory preview from recorded episode state traces.
- Metric cards populated from real episode metrics and agent diagnostics.

## Explicit Placeholders

The GUI intentionally shows `NaN` or `未完成` for capabilities whose runtime
path is not implemented yet:

- LE-WM training wizard.
- UE5 live bridge monitor.
- Live camera/depth sensor panels.
- Mid-episode pause, resume, and cancellation.

These placeholders should be replaced only when the corresponding backend or
model workflow is implemented in `offroad_sim`.
