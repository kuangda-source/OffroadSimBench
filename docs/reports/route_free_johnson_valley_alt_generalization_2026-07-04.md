# Johnson Valley Alternate Start/Goal Generalization - 2026-07-04

## Goal

Verify that the route-free BeamNG world-model workflow can train and evaluate a
new start/goal task on the same Johnson Valley map without changing Python code.

## New Task

- Task: `configs/tasks/beamng_johnson_valley_nav_alt.yaml`
- Map: `johnson_valley`
- Start: `(1281.0656738281, -110.71189117432, 115.508)`
- Goal: `(1233.2299804688, -165.27005004883)`
- Goal radius: `12.0 m`
- Expert route points: `7`
- Evaluation drive mode: `manual`
- BeamNG `ai_line` is only used for route-aware data collection and
  route-guided comparison, not for route-free model-control evaluation.

## Training Run

- Output:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260704`
- Model:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260704/model`
- Training record:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260704/training_run.json`
- Trajectory plot:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260704/region_self_supervised_trajectory.svg`

Settings:

- `world_model_type=mlp_dynamics`
- `collection_strategy=route_aware`
- `collect_rollouts=5`
- `collect_steps=900`
- `collection_multi_start=true`
- `collection_route_lateral_m=2.5`
- `collection_multi_start_lateral_m=1.25`
- `min_route_coverage_ratio=0.5`
- `min_goal_zone_coverage=0.2`
- `max_collection_min_goal_distance_m=70.0`
- `min_unique_region_cells=3`
- `use_experience_corridor=false`
- `evaluation_use_model_support_subgoals=true`
- `evaluation_allow_reverse_recovery=false`

## Collection Quality

- Quality gate: `passed`
- `collection_progress_ratio=0.8380706012542076`
- `route_coverage_ratio=1.0`
- `goal_zone_coverage=1.0`
- `collection_min_goal_distance=11.74947739790029`
- `collection_goal_reached=true`
- `unique_region_cells=4`
- `collection_collision_count=0`

## Model Quality

- `train_rmse=0.023343859292325984`
- `validation_rmse=0.024583964014261137`
- Segment sample counts:
  - `start=278`
  - `middle=354`
  - `goal=193`
- Segment RMSE:
  - `start=0.02788176403928973`
  - `middle=0.021096154283892637`
  - `goal=0.02107435448949935`

## Route-Free Evaluation

Initial self-supervised evaluation:

- `goal_success=true`
- `final_goal_distance=11.838430846480756`
- `min_goal_distance=11.838430846480756`
- `collision_count=0`
- `distance_traveled=88.09575819806722`
- `stuck_recovery_count=0`
- `reverse_count=0`

## Route-Free Vs Route-Guided Baseline

Comparison run:

- Summary:
  `outputs/region_world_model_eval/johnson_valley_alt_mlp_support_baseline_20260704/region_world_model_evaluation_summary.json`
- Trajectory plot:
  `outputs/region_world_model_eval/johnson_valley_alt_mlp_support_baseline_20260704/region_world_model_trajectory.svg`

Route-free model-control result:

- `route_free_goal_success=true`
- `route_free_final_goal_distance=11.945353031319677`
- `route_free_collision_count=0`
- `route_free_distance_traveled=87.38984768032832`
- `route_free_stuck_recovery_count=0`
- `route_free_reverse_count=0`

Route-guided comparison:

- `route_guided_goal_success=true`
- `route_guided_final_goal_distance=11.637721142937917`
- `route_guided_collision_count=0`
- `route_guided_distance_traveled=84.74289484715894`
- `route_guided_stuck_recovery_count=1`
- `route_guided_reverse_count=0`

## Interpretation

This validates the first same-map generalization target: a new Johnson Valley
task can be configured as data, then run through the existing route-aware
collection, MLP dynamics training, model-support subgoal evaluation, and
route-guided comparison without changing Python code.

The result still depends on a task expert route for data collection curriculum
and for the comparison baseline. The route-free evaluation itself receives the
start, goal, region, and model-owned support subgoals from the trained artifact;
it does not use BeamNG `ai_line` or task route waypoints during control.

## Remaining Risks

- This is one alternate start/goal on the same local region, not a broad
  Johnson Valley benchmark suite.
- The support-subgoal model is still a local navigation aid derived from
  collected trajectories, not a fully learned global world model.
- More random seeds and farther alternate tasks are needed before claiming broad
  generalization.
