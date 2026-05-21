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

- Left navigation is organized by workflow: overview, dataset, world model,
  path planning, BeamNG, and episode records.
- The overview page keeps only basic run configuration, the selected BeamNG
  region task, a saved world-model config selector, and the start button.
- The world model page owns checkpoint path, algorithm, and world-model type
  selection; saving a config makes it available to the overview dropdown.
- Runtime catalogs for backends, agents, world models, and planners.
- Dataset inspection through the registered dataset adapters.
- Episode execution through `offroad_sim.evaluation.run_episode`.
- Tiny learned world-model training through `TinyLearnedWorldModel.fit`.
- StableWM HDF5 export with configurable image size from the advanced settings dialog.
- LE-WM-compatible cost-model training from exported HDF5 files.
- One-click ORFD -> HDF5 -> LE-WM cost checkpoint -> dataset replay -> optional BeamNG run.
- ORFD frame preview for RGB, depth, and label assets, including official ORFD ZIP releases.
- ORFD-derived local heightmap/OBJ terrain draft export for BeamNG map prototyping.
- BeamNG navigation task editing with realtime preview of region, start/goal, route markers, camera height,
  draggable region points, delayed save-only validation warnings, translucent BeamNG region masks, and
  current-vehicle world-coordinate picking from the live BeamNG map.
- Episode list loading from `outputs/episodes`.
- BEV-style trajectory preview from recorded episode state traces.
- Metric cards populated from real episode metrics and agent diagnostics.
- Advanced settings dialog for planner samples/iterations, image export size, preview frame,
  terrain grid size, recording flags, and BeamNG pipeline behavior.

## Explicit Placeholders

The GUI intentionally shows `NaN` or `未完成` for capabilities whose runtime
path is not implemented yet:

- Full ORFD scene-level BeamNG level packaging.
- Full upstream LE-WM visual latent training.
- UE5 live bridge monitor.
- Mid-episode pause, resume, and cancellation.

These placeholders should be replaced only when the corresponding backend or
model workflow is implemented in `offroad_sim`.
