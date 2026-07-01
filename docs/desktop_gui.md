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
- The overview page is a guided demo launcher: select a single `Demo config`,
  click `Start demo`, and review the result summary. The selected demo config
  owns the BeamNG region task, saved world-model config, planner, and visible
  runtime defaults.
- The standard demo config is `johnson_valley_standard_demo`, backed by
  `configs/tasks/beamng_johnson_valley_nav_001.yaml`.
- The Dataset and Training workbench owns dataset import/preview, HDF5 export,
  checkpoint path, algorithm, world-model type, model training, and saved
  world-model configs.
- Dataset and Training can save reusable training configs that bind a dataset
  root/adapter/sequence, a built-in or imported trainer manifest, JSON training
  parameters, and a dedicated training output path.
- The Dataset tab can register a generic `manifest_dataset` directly from the
  current dataset root, display name, and JSON sequence/asset declarations via
  `Save dataset manifest`, so a new driving dataset does not require Python code
  before it can appear in the catalog.
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
- `Start training/export` runs the current training config through one service
  boundary. Imported `trainer.yaml` manifests can run external local model
  trainers and record `training_run.json`, metrics, history curves, stdout, and stderr.
  Trainers may return JSON on stdout or write sidecar files such as
  `metrics.json`, `history.json`, or `events.jsonl` in the output directory.
- Training run details show the recorded artifact, key metrics, available curve
  names, curve history, and stdout/stderr log paths for imported trainers.
- A completed training run with a runnable checkpoint or `model.json` artifact
  can be promoted directly from the Dataset and Training page with `Register
  latest training artifact`. The GUI writes a saved world-model config and
  records the promoted config back into `training_run.json`.
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
  collection-insufficient runs stay in Training Results only. Promoted config
  IDs and validation status are written back into `training_run.json`, so the
  Training Results tab can show which trained model is ready for BeamNG reuse.
  `goal_reached` means the vehicle entered the goal radius at least once;
  `goal_success` is stricter and also requires the final recorded pose to remain
  inside the goal radius and navigation region.
- Episode list loading from `outputs/episodes`.
- BEV-style trajectory preview from recorded episode state traces.
- Metric cards populated from real episode metrics and agent diagnostics.
- Advanced settings dialog for planner samples/iterations, image export size,
  preview frame, terrain grid size, and recording flags.

## External Training Bundle

For a quick end-to-end smoke test, select the built-in `Smoke tiny world model`
training config. It materializes a tiny ORFD-style dataset under
`outputs/training_studio_smoke/datasets/mock_orfd`, trains the local tiny world
model, writes `training_run.json`, and provides real metric history for the GUI
curves.

The standard demo can also be checked without opening the GUI:

```powershell
python scripts\demo_acceptance.py --demo-config johnson_valley_standard_demo --runs 1
```

The acceptance report is JSON and includes goal status, collision count, final
distance, trajectory length, average speed, and whether recovery logic was
triggered. Use `--runs 2` or `--runs 3` for repeatability checks.

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
stdout/stderr logs, metrics, and curve history. The trainer can emit a JSON
object on stdout, or write sidecar files in the output directory:

```text
metrics.json      # {"loss": 0.25, "accuracy": 0.75}
history.json      # {"loss": [0.9, 0.5, 0.25]}
events.jsonl      # {"step": 1, "loss": 0.9} per line
```

If a trainer does not ship with a manifest yet, the Model Training tab can
create one directly from a local script path. Set `Trainer entrypoint`, edit the
JSON `Trainer arguments` list such as `["{dataset_root}", "--output",
"{output_dir}"]`, optionally declare a JSON parameter schema, then click `Save
trainer from script`. For the faster path, fill the dataset fields, the script
path, and `Training parameters`, then click `Run script now`; the GUI will save
both the generated `trainer.yaml` and reusable training config before launching
the same unified training runner. If the argument template is still the default
dataset/output pair, the GUI infers command flags from the parameter names, for
example `{"epochs": 10}` becomes `--epochs {params.epochs}`. Training bundles
can also inline the same definition:

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
  outputs:
    artifact_type: checkpoint
    artifact_path: model.ckpt
    metrics_file: metrics.json
    history_file: history.json
    events_file: events.jsonl
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
