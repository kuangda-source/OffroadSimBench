# Johnson Valley Model-Support Route-Free Probe - 2026-07-04

## Goal

Move beyond the failing strict-direct route-free baseline by letting the model
own a learned support signal. The evaluation still removes BeamNG route metadata
and task expert route metadata from the scenario; it does not use BeamNG
`ai_line` during evaluation.

## Code Changes

- `TinyLearnedWorldModel` stores downsampled training `support_points` and adds
  support-distance risk to prediction metadata.
- `WorldModelDirectAgent` can optionally use model `support_points` as a
  local-subgoal route through `use_model_support_subgoals=True`.
- `RegionWorldModelEvaluationRequest` accepts planner weight overrides and the
  support-subgoal flag.
- GUI and demo acceptance paths now read validated world-model config settings:
  planner weights, local subgoal distance, reverse-recovery settings,
  experience-corridor flag, and model-support-subgoal flag.

## BeamNG Evidence

Task:
`configs/tasks/beamng_johnson_valley_nav_test.yaml`

Initial flat-support model:
`outputs/region_self_supervised/p2_support_risk_direct_20260704_022138/model`

Successful run:
`outputs/region_world_model_eval/p2_model_support_reverse_1200_20260704_024552/region_world_model_evaluation_summary.json`

Episode:
`outputs/episodes/beamng_johnson_valley_nav_test_evaluation_world_model_direct_20260703T184552Z`

Parameters:

- `evaluation_agent=world_model_direct`
- `use_experience_corridor=False`
- `evaluation_use_model_support_subgoals=True`
- `evaluation_local_subgoal_distance_m=16.0`
- `planner_horizon=8`
- `planner_samples=64`
- `planner_iterations=3`
- `planner_goal_weight=0.25`
- `planner_progress_weight=0.4`
- `planner_risk_weight=12.0`
- `planner_heading_weight=0.25`
- `evaluation_allow_reverse_recovery=True`
- `evaluation_reverse_recovery_after_steps=144`

Acceptance:

- `goal_success=true`
- `final_goal_distance=11.769145127352756`
- `min_goal_distance=11.769145127352756`
- `goal_radius=12.0`
- `route_waypoint_count=0`
- `drive_mode=manual`
- `collision_count=0`
- `distance_traveled=176.7101023334125`
- `stuck_recovery_count=0`
- `reverse_count=0`

## Comparison

- Strict direct without experience/support remained near the old failure mode:
  about `108 m` to `110 m` from the goal, depending on probe.
- Model support subgoals improved a 420-step probe to `81.491 m` without
  collisions or stuck recovery.
- The validated 1200-step support-subgoal probe reached the goal radius.

## Segmented Support Route Retest

The first implementation stored all rollout states as one flat
`support_points` sequence. That reached the goal, but it could silently bridge
the tail of one collection rollout into the head of another. The follow-up
change stores per-rollout `support_routes` and keeps the flat `support_points`
only for legacy risk scoring.

Naively selecting the nearest support route segment failed:

- Run:
  `outputs/region_world_model_eval/p2_support_routes_reverse_1200_20260704_031554/region_world_model_evaluation_summary.json`
- Final/minimum goal distance: `121.845 m` / `108.391 m`
- `stuck_recovery_count=660`, `reverse_count=132`
- Root cause: the agent stayed on segment 0 and kept targeting that segment's
  end point instead of advancing to the next rollout segment.

The corrected version builds an explicit support-route graph at runtime. It
starts from the nearest support route, then bridges only to a next segment whose
start is within a bounded gap and whose end is closer to the final goal. Large
gaps remain unbridged, so unrelated rollouts are not stitched together.

Validated segmented-support run:

- Model:
  `outputs/region_self_supervised/p2_support_routes_direct_20260704_031554/model`
- Summary:
  `outputs/region_world_model_eval/p2_support_routes_bridged_reverse_1200_20260704_032050/region_world_model_evaluation_summary.json`
- `support_route_count=4`
- `goal_success=true`
- `final_goal_distance=11.807668899363438`
- `min_goal_distance=11.807668899363438`
- `goal_radius=12.0`
- `route_waypoint_count=0`
- `drive_mode=manual`
- `collision_count=0`
- `distance_traveled=176.40412352208287`
- `stuck_recovery_count=0`
- `reverse_count=0`

Mean support-risk retest:

- Summary:
  `outputs/region_world_model_eval/p2_support_risk_mean_support_route_1200_20260704/region_world_model_evaluation_summary.json`
- `goal_success=true`
- `final_goal_distance=11.070667805902877`
- `min_goal_distance=11.070667805902877`
- `collision_count=0`
- `distance_traveled=179.3240986832698`
- `stuck_recovery_count=0`
- `reverse_count=0`

## Strict Collection MLP Retest And Default Demo Promotion

After the stricter route-aware collection gate was added, a new collection was
captured with full route coverage and goal-zone coverage:

- Collection manifest:
  `outputs/beamng_training_acceptance/route_aware_strict_20260704_collect/region_training_collection.json`
- `route_coverage_ratio=1.0`
- `goal_zone_coverage=1.0`
- `collection_min_goal_distance=11.698582123429976`
- `unique_region_cells=8`
- `collision_count=0`

The same collection trained both `tiny_learned` and `mlp_dynamics`. Strict
direct route-free still failed near the start, but `mlp_dynamics` with
model-owned support subgoals reached the goal without using task route waypoints:

- Model:
  `outputs/region_world_model_compare/johnson_valley_strict_collect_tiny_vs_mlp_1200_20260704/mlp_dynamics/training/model`
- Summary:
  `outputs/region_world_model_eval/p2_mlp_support_subgoals_strict_1200_20260704/region_world_model_evaluation_summary.json`
- `goal_success=true`
- `final_goal_distance=11.897536452288634`
- `min_goal_distance=11.897536452288634`
- `goal_radius=12.0`
- `route_waypoint_count=0`
- `collision_count=0`
- `distance_traveled=193.88469705904217`
- `stuck_recovery_count=0`
- `reverse_count=0`
- Route-guided comparison also succeeded with
  `final_goal_distance=11.599028573749857`.

The desktop default demo config now points at this validated MLP support-route
configuration so the first GUI demo exercises route-free model control rather
than the older LE-WM route-guided path.

## Interpretation

This is not a pure "only start, goal, and polygon" strict-direct success.
However, it is also not a hard-coded expert route in the scenario. The local
subgoals come from the trained model artifact's support points, so the behavior
is a model-owned navigation aid and can be swapped with future learned
traversability, latent map, or policy models.

## Remaining Risks

- The strict-direct baseline still fails and remains a useful diagnostic gate.
- The support route is now segmented per rollout and bridged with distance and
  goal-progress constraints, but it still represents a model-owned navigation
  aid rather than a fully learned route-free policy.
- The MLP support-route result is now the default demo entry, but stability still
  needs repeated multi-seed BeamNG validation.
- Generalization to a new Johnson Valley start/goal is not yet proven.
