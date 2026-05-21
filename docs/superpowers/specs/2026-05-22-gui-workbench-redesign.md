# OffroadSimBench GUI Workbench Redesign

Date: 2026-05-22

## Goal

Reorganize the PySide6 desktop GUI around two clear product areas:

1. **Dataset and Training Workbench**: operate on datasets, train or load
   algorithms/models, and inspect training or inference results.
2. **BeamNG Simulation Workbench**: configure maps/tasks, select trained model
   configs, run visible autonomous driving, and evaluate simulation results.

The overview page becomes a guided demo launcher for new users. It should guide
users through the minimum configuration required to run a visible demo, then show
the demo outcome. It should not expose detailed training, model, or BeamNG task
editing controls.

## Current State

The code already has the core runtime pieces needed for this direction:

- `AlgorithmAdapter` and `algorithm.yaml` support pluggable algorithm packages.
- `local_lewm_cost` can prepare data, train a local LE-WM-compatible cost model,
  and score action candidates.
- `stablewm_lewm` can load an existing checkpoint and score actions.
- The GUI can save a world-model config containing checkpoint path, algorithm,
  and world-model type.
- BeamNG region task editing, realtime preview, model-controlled region
  navigation, and episode metrics are already connected.

The GUI is the weak point: dataset operations, model training, model config, and
BeamNG simulation are spread across pages in a way that makes the workflow feel
like a stack of controls instead of a product.

## Information Architecture

The left navigation should be reorganized into:

- **Overview**
  - Guided demo launcher
  - Environment status
  - Demo result summary
  - Links to the two workbenches
- **Dataset and Training**
  - Dataset import and preview
  - Dataset processing and labels
  - Model/algorithm training
  - Inference and training results
- **BeamNG Simulation**
  - Map and task selection
  - Region/start/goal/route editor
  - Simulation model config
  - Autodrive run and evaluation
- **Experiment Records**
  - Episodes, trajectories, metrics, and logs

This can be implemented with the current `QStackedWidget` shell, but each major
workbench should own its own internal tabs or segmented navigation. A full GUI
rewrite is not required unless the current main-window file becomes too hard to
maintain during implementation.

## Overview Page Design

The overview page is a guided demo launcher:

1. **Choose demo preset**
   - Default: Johnson Valley LE-WM navigation.
   - Later: ORFD dataset preview or other curated demos.
2. **Check requirements**
   - BeamNG status.
   - Dataset path status when the preset needs a dataset.
   - Default checkpoint/model config status.
   - Python runtime status.
3. **Confirm minimum config**
   - BeamNG region task.
   - World model config.
   - Planner.
4. **Run and show result**
   - Start the selected demo.
   - Show real metrics only after a run: goal success, minimum goal distance,
     collision count, route progress, and episode path.
   - Before a run, show `NaN` or unfinished status instead of placeholder
     success values.

The page should include clear entry buttons for:

- Open Dataset and Training Workbench.
- Open BeamNG Simulation Workbench.
- Open Experiment Records.

Detailed model paths, algorithm internals, training controls, region editing,
and advanced BeamNG parameters should not live on the overview page.

## Dataset and Training Workbench

This workbench should answer: "What data do I have, what can I train, and how
did the model perform on dataset-side checks?"

Recommended internal sections:

- **Dataset Import and Preview**
  - Dataset root, adapter, sequence selection.
  - ORFD image preview and asset inspection.
  - Existing HDF5 export controls.
- **Dataset Processing and Labels**
  - Segmentation, label, terrain mask, or preprocessing previews.
  - Initially show unavailable items as `NaN` or unfinished.
- **Model/Algorithm Training**
  - Select algorithm adapter.
  - Select dataset or prepared HDF5 input.
  - Configure training output path and core training parameters.
  - Run training through the adapter capability when supported.
- **Inference and Results**
  - Training logs and artifacts.
  - Checkpoint list.
  - Dataset-side inference preview and metrics.
  - Save trained output as a reusable world-model config.

The existing one-click `ORFD -> LE-WM -> BeamNG` button should be removed from
the world-model page. It crosses both major workbenches and should later return
only as an explicit pipeline/demo preset if the intermediate steps are visible.

## BeamNG Simulation Workbench

This workbench should answer: "What BeamNG task am I running, which model is
driving, and how did it perform?"

Recommended internal sections:

- **Map and Task**
  - BeamNG level/map selection.
  - Region task selection.
  - Task validation status.
- **Region Editor**
  - Reuse the existing non-modal realtime editor.
  - Keep BeamNG click picking, draggable region points, right-click deletion,
    camera height/mode controls, and delayed validation warnings.
- **Simulation Model Config**
  - Select saved world-model config.
  - Show checkpoint path, algorithm, and world-model type as read-only summary
    by default, with an edit link into the training/config section.
- **Run and Evaluation**
  - Run model-controlled BeamNG navigation.
  - Show metrics: goal success, minimum/final goal distance, collision count,
    route progress, out-of-region status, mean throttle, and mean absolute steer.

## External Model Training and Use

The system should not require hard-coded GUI changes for every new model.

The intended flow for another model or algorithm is:

1. A developer creates an algorithm package with `algorithm.yaml`.
2. The package implements the relevant `AlgorithmAdapter` capabilities:
   `prepare_data`, `train`, `load`, `score_actions`, or future inference hooks.
3. The Dataset and Training workbench discovers the adapter and shows supported
   actions.
4. Training output is saved as a world-model config.
5. The BeamNG Simulation workbench uses that config to run autonomous driving.

Today this is partly implemented at the backend layer, but the GUI still needs a
proper adapter-driven training form and results page. Until then, arbitrary
GitHub models still need a small adapter wrapper before they become first-class
GUI training options.

## Error Handling and Placeholders

- Missing datasets, checkpoints, adapters, or BeamNG should be visible in status
  cards with a concrete message.
- Unimplemented items should display `NaN` or unfinished status.
- No fake metrics should be shown.
- Long-running training and BeamNG operations must run in worker threads so the
  GUI remains responsive.

## Testing Strategy

Add or update tests for:

- Overview page no longer exposes detailed model training controls.
- Overview presets can resolve their required task and world-model config.
- Dataset and Training workbench exposes dataset preview and training controls.
- BeamNG workbench exposes task, region editor, model config, run, and evaluation
  controls.
- Saved world-model configs are readable from both overview and BeamNG pages.
- Removing the one-click ORFD-to-BeamNG button does not remove the underlying
  service functions or tests.

Existing validation commands remain:

```powershell
python -m pytest -q
python examples/run_gym_demo.py --agent rule_based --max-steps 1200
python -m offroad_sim.cli list
```

For GUI changes:

```powershell
python -m pytest tests/test_desktop_services.py tests/test_desktop_visible_demo.py -q
$env:QT_QPA_PLATFORM='offscreen'; python -c "from PySide6.QtWidgets import QApplication; from desktop_app.qt_main import MainWindow; app=QApplication([]); w=MainWindow(); print(w.windowTitle())"
```

## Implementation Order

1. Remove the one-click ORFD-to-LE-WM-to-BeamNG button from the world-model page.
2. Rename/restructure navigation into overview, dataset/training, BeamNG
   simulation, and records.
3. Convert overview into guided demo launcher using existing task and model
   config services.
4. Move training and result controls into the Dataset and Training workbench.
5. Move BeamNG model config and run/evaluation controls into the BeamNG
   Simulation workbench.
6. Add tests for the new information architecture.

## Deferred Scope

- Full dataset-to-BeamNG map reconstruction remains deferred.
- Full external-model no-code onboarding remains deferred, but the adapter-based
  route must be preserved.
- UI visual polish can iterate after the information architecture is stable.
