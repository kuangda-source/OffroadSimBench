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

- Left navigation is organized by workflow: guided overview, Dataset and
  Training, BeamNG Simulation, and episode records.
- The overview page is a guided demo launcher: select a demo preset, confirm
  the BeamNG region task, world-model config, and planner, then run the visible
  demo.
- The Dataset and Training workbench owns dataset import/preview, HDF5 export,
  checkpoint path, algorithm, world-model type, model training, and saved
  world-model configs.
- Dataset and Training can save reusable training configs that bind a dataset
  root/adapter/sequence, a built-in or imported trainer manifest, JSON training
  parameters, and a dedicated training output path.
- The BeamNG Simulation workbench owns task selection, region/start/goal
  editing, model-config selection, runtime checks, simulation runs, evaluation,
  and terrain draft export.
- Runtime catalogs for backends, agents, world models, and planners.
- Dataset inspection through the registered dataset adapters.
- Episode execution through `offroad_sim.evaluation.run_episode`.
- Tiny learned world-model training through `TinyLearnedWorldModel.fit`.
- StableWM HDF5 export with configurable image size from the advanced settings dialog.
- LE-WM-compatible cost-model training from exported HDF5 files.
- Imported `trainer.yaml` manifests can run external local model trainers and
  record `training_run.json`, metrics, history curves, stdout, and stderr.
- Existing model checkpoints, lightweight `model.json` files, or model folders
  containing `model.json` can be imported as saved world-model configs, then
  reused by the overview launcher and BeamNG simulation page without editing
  JSON by hand.
- ORFD frame preview for RGB, depth, and label assets, including official ORFD ZIP releases.
- ORFD-derived local heightmap/OBJ terrain draft export for BeamNG map prototyping.
- BeamNG navigation task editing with realtime preview of region, start/goal, route markers, camera height,
  draggable region points, delayed save-only validation warnings, translucent BeamNG region masks, and
  current-vehicle world-coordinate picking from the live BeamNG map.
- Episode list loading from `outputs/episodes`.
- BEV-style trajectory preview from recorded episode state traces.
- Metric cards populated from real episode metrics and agent diagnostics.
- Advanced settings dialog for planner samples/iterations, image export size,
  preview frame, terrain grid size, and recording flags.

## Explicit Placeholders

The GUI intentionally shows `NaN` or `未完成` for capabilities whose runtime
path is not implemented yet:

- Full ORFD scene-level BeamNG level packaging.
- Full upstream LE-WM visual latent training.
- UE5 live bridge monitor.
- Mid-episode pause, resume, and cancellation.

These placeholders should be replaced only when the corresponding backend or
model workflow is implemented in `offroad_sim`.
