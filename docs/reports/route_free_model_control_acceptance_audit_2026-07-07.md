# Route-Free Model Control Acceptance Audit - 2026-07-07

This audit checks the active P0-P5 objective against the current codebase and
the BeamNG Johnson Valley evidence available in `outputs/`. It treats completion
as unproven unless there is code, test, or runtime-output evidence.

## Verdict

The current route-free model-control milestone is accepted for the
`model_support_graph_subgoals` mode. The runtime evaluation does not inject the
task expert route into the route-free scenario: the route-free scenario uses the
start, goal, region, and a sparse support graph saved inside the learned model
checkpoint. That is stronger than the older ordered support-route demo, but it is
not yet a pure perception-only world model. Strict direct control without learned
topology remains a research target and is kept as a diagnostic lower bound.

## P0 - Fixed Baselines and Metrics

Implemented and verified.

- `desktop_app/services.py` defines `RegionWorldModelEvaluationRequest` with
  `include_route_guided_baseline`.
- `_run_region_world_model_evaluation()` always runs the route-free scenario and
  can also run the route-guided baseline.
- `_route_free_region_scenario()` removes route waypoints from the evaluation
  scenario. The Johnson Valley support-graph summaries show route-free
  evaluation with zero route waypoints while the route-guided baseline keeps the
  route.
- `_region_world_model_comparison()` records goal success, minimum/final goal
  distance, collision count, distance traveled, stuck recovery count, and reverse
  count for both baselines.
- `_write_region_trajectory_svg()` writes the region, start, goal, expert route,
  collection/evaluation traces, and now text labels for `start` and `goal`.
- Tests:
  `test_region_world_model_evaluation_compares_route_free_and_route_guided_baselines`
  checks the comparison metrics and SVG labels.

Runtime evidence:

| Task | Route-free success | Route-free final/min distance | Collisions | Distance | Stuck | Reverse | Route-guided success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `beamng_johnson_valley_nav_test` | true | 11.988 m | 0 | 194.965 m | 0 | 0 | true |
| `beamng_johnson_valley_nav_alt` | true | 11.837 m | 0 | 81.780 m | 0 | 0 | true |

Evidence files:

- `outputs/region_world_model_eval/johnson_valley_mlp_support_graph_20260707_live/region_world_model_evaluation_summary.json`
- `outputs/region_world_model_eval/johnson_valley_alt_mlp_support_graph_20260707_live/region_world_model_evaluation_summary.json`
- `outputs/region_world_model_eval/johnson_valley_mlp_support_graph_20260707_live/region_world_model_trajectory.svg`
- `outputs/region_world_model_eval/johnson_valley_alt_mlp_support_graph_20260707_live/region_world_model_trajectory.svg`

## P1 - Route-Aware Curriculum Collection and Quality Gates

Implemented and verified.

- `RegionTrainingDataCollectionRequest` and
  `RegionSelfSupervisedWorldModelRequest` expose route-aware collection,
  multi-start collection, lateral waypoint perturbation, and quality thresholds.
- `_route_aware_collection_options()` and `_apply_route_multi_start()` generate
  route-segment collection targets instead of only sampling near the initial
  start.
- `_collection_quality_gate()` blocks weak collection manifests when route
  coverage, goal-zone coverage, collection minimum goal distance, or unique
  region cells fail the configured thresholds.
- Tests cover route-aware collection success plus failure modes for route
  coverage, goal-zone coverage, collection minimum goal distance, and unique
  region cells.

Runtime collection evidence:

| Collection | Route coverage | Goal-zone coverage | Min goal distance | Unique cells |
| --- | ---: | ---: | ---: | ---: |
| current task | 1.0 | 1.0 | 11.699 m | 8 |
| alternate task | 1.0 | 1.0 | 11.877 m | 4 |

Evidence files:

- `outputs/beamng_training_acceptance/route_aware_strict_20260704_collect/region_training_collection.json`
- `outputs/beamng_training_acceptance/alt_support_graph_20260707_collect/region_training_collection.json`

## P2 - Route-Free Controller Improvements

Implemented for the accepted support-graph mode; strict direct is still not
accepted.

- `world_model_direct` now supports dynamic local subgoals, experience corridor
  penalties, and model-owned support-graph subgoals.
- Reverse recovery is treated as a last-resort diagnostic; the accepted
  support-graph evaluations both report zero reverse steps and zero stuck
  recoveries.
- The planner cost includes target progress, heading alignment, low-speed/stuck
  behavior, support evidence, and region-boundary costs.
- The accepted mode uses `model_support_graph_subgoal_used=true` and
  `experience_corridor_used=false`.

Runtime evidence:

- Current task: route-free support graph reached the 12 m goal radius with
  `collision_count=0`, `stuck_recovery_count=0`, and `reverse_count=0`.
- Alternate task: same result pattern on a different start/goal task without code
  changes.

## P3 - Model Training Upgrade

Implemented and verified for `tiny_learned` and `mlp_dynamics`; the accepted
checkpoint uses `mlp_dynamics`.

- Training records include train RMSE, validation RMSE, segment RMSE, segment
  sample counts, and history arrays for GUI display.
- `tiny_learned` and `mlp_dynamics` checkpoints save a sparse support graph from
  collected trajectories.
- Tests check validation RMSE, segment RMSE, segment sample counts, and
  navigation-readiness diagnostics.

Runtime training evidence:

| Checkpoint | Train RMSE | Validation RMSE | Segment counts | Support graph |
| --- | ---: | ---: | --- | --- |
| `johnson_valley_mlp_support_graph_20260707` | 0.020760 | 0.021027 | start 777, middle 820, goal 435 | 954 nodes / 948 edges |
| `johnson_valley_alt_mlp_support_graph_20260707` | 0.013401 | 0.013653 | start 261, middle 347, goal 198 | 732 nodes / 727 edges |

Evidence files:

- `outputs/beamng_region_world_models/johnson_valley_mlp_support_graph_20260707/region_world_model_training_summary.json`
- `outputs/beamng_region_world_models/johnson_valley_alt_mlp_support_graph_20260707/region_world_model_training_summary.json`

## P4 - GUI Workflow

Implemented at the service/GUI integration level and covered by tests.

- The BeamNG page is split into collection, training, and evaluation actions.
- The world-model/demo configuration validation only marks a route-free model as
  demo-ready when it is explicitly labeled and has evidence such as zero route
  waypoints, accepted route-free mode, and successful validation.
- GUI report fields include route-free/route-guided metrics, collection quality,
  validation RMSE, and segment metrics.
- Tests cover GUI request construction for collection, training, and evaluation
  modes.

Primary files:

- `desktop_app/qt_main.py`
- `desktop_app/services.py`
- `tests/test_desktop_layout.py`
- `tests/test_desktop_services.py`
- `tests/test_region_self_supervised_world_model.py`

## P5 - Acceptance

Accepted for support-graph route-free model control.

1. First-stage target: route-guided baseline stable success.
   - Current and alternate Johnson Valley route-guided baselines both succeed.
2. First-stage target: route-free clearly improves beyond the old direct failure.
   - Strict direct remained around 112-115 m in the documented probe.
   - Support-graph route-free reached 11.988 m on the current task and 11.837 m
     on the alternate task.
3. Second-stage target: route-free reaches the Johnson Valley current task goal
   without collision.
   - Current task succeeded with 0 collisions.
4. Third-stage target: a same-map new start/goal can train/evaluate without code
   edits.
   - `configs/tasks/beamng_johnson_valley_nav_alt.yaml` used the same
     collection/training/evaluation code path and succeeded with 0 collisions.

## Remaining Research Gap

The accepted configuration is route-free at evaluation time but still benefits
from route-aware training data and the learned support graph. The next harder
research milestone is either strict direct control from only local model costs or
a learned traversability/frontier policy that can discover useful topology from
broader exploration instead of route-aware collection.
