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
- A complete external training bundle can be imported through
  `Import training config`. The bundle config may reference a
  `dataset_manifest.yaml`, a `trainer.yaml`, an output path, and parameter
  values; the GUI installs the referenced dataset/trainer manifests and selects
  the saved training config automatically.
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
- Training run details show the recorded artifact, key metrics, curve history,
  and stdout/stderr log paths for imported trainers.
- Existing model checkpoints, lightweight `model.json` files, or model folders
  containing `model.json` can be imported as saved world-model configs, then
  reused by the overview launcher and BeamNG simulation page without editing
  JSON by hand.
- ORFD frame preview for RGB, depth, and label assets, including official ORFD ZIP releases.
- ORFD-derived local heightmap/OBJ terrain draft export for BeamNG map prototyping.
- BeamNG navigation task editing with realtime preview of region, start/goal, route markers, camera height,
  draggable region points, delayed save-only validation warnings, translucent BeamNG region masks, and
  current-vehicle world-coordinate picking from the live BeamNG map.
- Region self-supervised and direct world-model runs report a compact BeamNG
  acceptance summary with goal success, goal distance, collision count,
  model-controlled status, quality-gate progress, and artifact paths.
  Successful self-supervised runs that reach `goal_success` are also promoted
  to saved world-model configs with validation metadata; failed or
  collection-insufficient runs stay in Training Results only.
  `goal_reached` means the vehicle entered the goal radius at least once;
  `goal_success` is stricter and also requires the final recorded pose to remain
  inside the goal radius and navigation region.
- Episode list loading from `outputs/episodes`.
- BEV-style trajectory preview from recorded episode state traces.
- Metric cards populated from real episode metrics and agent diagnostics.
- Advanced settings dialog for planner samples/iterations, image export size,
  preview frame, terrain grid size, and recording flags.

## External Training Bundle

For a new autonomous-driving dataset or model trainer, the preferred handoff is
three small files. Put paths relative to `training_config.yaml` so the bundle can
be moved as one folder:

```yaml
# training_config.yaml
id: my_drive_experiment
label: My Drive Experiment
dataset_manifest: dataset/dataset_manifest.yaml
trainer_manifest: trainer/trainer.yaml
sequence_id: clip_001
output_path: outputs/models/my_drive_experiment
parameters:
  epochs: 10
  batch_size: 8
```

`dataset_manifest.yaml` describes the dataset layout through the
`manifest_dataset` adapter. `trainer.yaml` describes the executable entrypoint,
command arguments, declared parameters, and artifact type. After importing the
training config, use `Start training/export`; the run writes `training_run.json`,
stdout/stderr logs, metrics, and curve history when the trainer emits JSON.

If a trainer does not ship with a manifest yet, the Model Training tab can
create one directly from a local script path. Set `Trainer entrypoint`, edit the
JSON `Trainer arguments` list such as `["{dataset_root}", "--output",
"{output_dir}"]`, optionally declare a JSON parameter schema, then click `Save
trainer from script`. Training bundles can also inline the same definition:

```yaml
trainer:
  trainer_id: inline_trainer
  display_name: Inline Trainer
  runtime: python
  entrypoint: train.py
  arguments:
    - "{dataset_root}"
    - "--output"
    - "{output_dir}"
  parameters:
    epochs:
      type: int
      default: 10
```

## Explicit Placeholders

The GUI intentionally shows `NaN` or `未完成` for capabilities whose runtime
path is not implemented yet:

- Full ORFD scene-level BeamNG level packaging.
- Full upstream LE-WM visual latent training.
- UE5 live bridge monitor.
- Mid-episode pause, resume, and cancellation.

These placeholders should be replaced only when the corresponding backend or
model workflow is implemented in `offroad_sim`.
