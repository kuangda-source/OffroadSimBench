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
configuration must provide a split JSON file.

Supported argument placeholders are:

- `{dataset_root}`
- `{adapter}`
- `{sequence_id}`
- `{split_path}`
- `{output_dir}`
- `{manifest_dir}`
- `{params.<name>}`

## Parameter Schema

Parameter types are `str`, `path`, `int`, `float`, and `bool`. A parameter can
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

## Checkpoint Inference

A trainer can add an optional `inference` section with its own `launch`,
`input`, `parameters`, `arguments`, and `outputs` fields. It uses the same three
launch kinds and parameter rules as training. In addition to the common dataset
placeholders, inference commands receive `{artifact_path}` and
`{checkpoint_path}`.

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

The workbench catalogs completed training artifacts, reports whether their
source trainer supports inference, validates the artifact and dataset contract,
and writes each result to `inference_run.json`.

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
