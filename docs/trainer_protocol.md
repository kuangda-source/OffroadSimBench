# Trainer Manifest Protocol

OffroadSimBench discovers versioned trainer manifests from
`configs/trainers/*.yaml`, `configs/trainers/*/trainer.yaml`, and imported local
manifests. Adding a trainer does not require changes to platform Python code.

Start from `configs/trainers/trainer.template.yaml` and set
`schema_version: 1`. The desktop workbench turns the parameter schema into
typed controls, validates the selected dataset, previews the command, and
records stdout, stderr, metrics, history, and artifacts in `training_run.json`.

## Launch Kinds

- `python_script`: runs the configured `entrypoint` with the environment Python.
- `python_module`: runs `python -m <module>`; set `launch.module` instead of an
  entrypoint.
- `executable`: runs a native executable or command directly.

`launch.conda_env` accepts either a Conda environment name or an environment
prefix. `launch.working_directory`, script paths, and executable paths are
resolved relative to the manifest. `launch.environment` is merged into the
process environment and supports `{manifest_dir}` and `{pathsep}` placeholders.

## Dataset Contract

`input.dataset_format` accepts `any_registered_adapter`, one adapter id, or a
list of adapter ids. `required_modalities` triggers a preflight check against a
normalized dataset sequence. Set `split_required: true` when the training
configuration must provide a split JSON file. The workbench validates that the
split belongs to the selected dataset and adapter, passes it through
`{split_path}`, and records the resolved path in `training_run.json`.
Validation also rejects duplicate samples and train/validation/test overlap.
Each run stores immutable manifest and split snapshots with SHA-256 hashes;
reruns use those snapshots instead of silently consuming edited source files.
Snapshot hashes and the referenced trainer entrypoint hash are verified before
a historical rerun starts; edited inputs fail fast instead of changing results.

Supported argument placeholders are:

- `{dataset_root}`
- `{adapter}`
- `{sequence_id}`
- `{split_path}`
- `{output_dir}`
- `{manifest_dir}`
- `{params.<name>}`

## Parameter Schema

Parameter types are `str`, `path`, `file`, `directory`, `int`, `float`, and
`bool`. Path, file, and directory parameters receive a GUI browse control. A parameter can
declare `default`, `required`, `min`, `max`, `step`, `enum`, `description`, and
`depends_on`. Dependencies accept a parameter name or a mapping such as:

```yaml
depends_on:
  parameter: use_amp
  equals: true
```

The trainer should write artifacts under `{output_dir}` and print one JSON
object containing optional `artifact_path`, `artifact_type`, `metrics`, and
`history` fields. The existing sidecar metrics and JSONL event formats remain
supported.

## Live Metrics

For live monitoring, write one flat or nested JSON object per line to
`{output_dir}/events.jsonl` (or declare another name in
`outputs.events_file`). Numeric fields are accepted as custom metrics without
platform changes. Common examples are:

```json
{"step": 12, "train_loss": 0.42, "validation_loss": 0.51, "learning_rate": 0.0003, "throughput": 38.2}
```

Each metric keeps the `global_step`, `step`, `iteration`, or `epoch` from the
event that produced it. Sparse validation metrics therefore remain aligned with
dense training metrics. Final JSON output may provide the same mapping directly
as `history_steps`, or through `outputs.history_steps_file`; otherwise the
workbench uses zero-based point indices.

The desktop workbench updates these series while the process is running,
automatically overlays train and validation loss, and exposes every other
numeric field in the metric selector. Managed jobs add `resource.cpu_percent`,
`resource.memory_mb`, and, when `nvidia-smi` sees the trainer process,
`resource.gpu_percent` and `resource.gpu_memory_mb`. Resource sampling uses the
optional `psutil` GUI dependency and never prevents a trainer from running.

Completed records are checked for non-finite values, loss explosions, stalled
series, and inactive event streams. The Training Results page exports normalized
metrics to JSON and CSV together with the currently visible PNG plot.

## Checkpoint Inference

A trainer can add an optional `inference` section with its own `launch`,
`input`, `parameters`, `arguments`, and `outputs` fields. It uses the same three
launch kinds and parameter rules as training. In addition to the common dataset
placeholders, inference commands receive `{artifact_path}` and
`{checkpoint_path}`. Inference may also set `input.split_required: true` and use
`{split_path}`. The Training Results page exposes inference parameters as JSON,
so batch size or a `validation`/`test` split selector can be changed without a
command line.

The inference process may print `metrics`, `history`, `predictions`, and
`previews` in its final JSON object. It may instead declare sidecars:

```yaml
inference:
  launch:
    kind: python_script
    entrypoint: infer.py
  arguments:
    - --checkpoint
    - "{artifact_path}"
    - --dataset
    - "{dataset_root}"
    - --output
    - "{output_dir}"
  outputs:
    predictions_file: predictions.json
    metrics_file: metrics.json
    preview_file: preview.png
```

The same mapping can live in a standalone `inference.yaml`. Reference it from
the trainer, or place it next to `trainer.yaml` for automatic discovery:

```yaml
# trainer.yaml
schema_version: 1
trainer_id: my_model
inference_manifest: inference.yaml
```

```yaml
# inference.yaml
schema_version: 1
inference_id: my_model
launch:
  kind: python_script
  entrypoint: infer.py
parameters:
  split_name:
    type: str
    default: validation
    enum: [validation, test]
```

Imported trainers embed the resolved inference contract so moving the installed
trainer manifest does not invalidate relative inference paths.

The workbench catalogs completed training artifacts, reports whether their
source trainer supports inference, validates the artifact and dataset contract,
and writes each result to `inference_run.json`.
Training is marked complete only when the declared artifact exists. Inference
subprocess and post-processing failures both persist parameters, split
provenance, logs, and the error in a failed `inference_run.json`.

## Minimal Module Trainer

```yaml
schema_version: 1
trainer_id: example_module
display_name: Example Module
launch:
  kind: python_module
  module: package.train
  conda_env: my-env
  working_directory: ../model-repository
  environment:
    PYTHONPATH: "{manifest_dir}{pathsep}../model-repository"
arguments:
  - --dataset
  - "{dataset_root}"
  - --output
  - "{output_dir}"
outputs:
  artifact_type: checkpoint
```
