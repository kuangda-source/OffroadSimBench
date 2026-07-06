# Johnson Valley Alternate Start/Goal Generalization - 2026-07-07

## Goal

Verify again, with the current code, that a same-map alternate Johnson Valley
start/goal task can be trained and evaluated without Python code changes.

## Task

- Task: `configs/tasks/beamng_johnson_valley_nav_alt.yaml`
- Map: `johnson_valley`
- Start: `(1281.0656738281, -110.71189117432, 115.508)`
- Goal: `(1233.2299804688, -165.27005004883)`
- Goal radius: `12.0 m`
- Expert route points: `7`
- Evaluation drive mode: `manual`

## Fresh Training Run

- Output:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260706_live`
- Model:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260706_live/model`
- Training record:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260706_live/training_run.json`
- Summary:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260706_live/region_self_supervised_summary.json`
- Trajectory plot:
  `outputs/region_self_supervised/johnson_valley_alt_mlp_generalization_20260706_live/region_self_supervised_trajectory.svg`

Command shape:

```powershell
python scripts\run_region_self_supervised_world_model.py configs\tasks\beamng_johnson_valley_nav_alt.yaml --world-model-type mlp_dynamics --collect-steps 900 --collect-rollouts 5 --collection-strategy route_aware --collection-multi-start --no-experience-corridor --evaluation-use-model-support-subgoals --evaluation-local-subgoal-distance-m 12
```

## Collection Quality

- Quality gate: `passed`
- `collection_progress_ratio=0.8363064850795001`
- `route_coverage_ratio=1.0`
- `goal_zone_coverage=1.0`
- `collection_min_goal_distance=11.877480362664814`
- `collection_goal_reached=true`
- `unique_region_cells=4`
- `collection_collision_count=0`

## Model Quality

- `world_model_type=mlp_dynamics`
- `model_family=random_feature_mlp`
- `sample_count=806`
- `train_rmse=0.01340104433141595`
- `validation_rmse=0.013653330250098227`
- Segment sample counts:
  - `start=261`
  - `middle=347`
  - `goal=198`
- Segment RMSE:
  - `start=0.011842852403592233`
  - `middle=0.01390475833711922`
  - `goal=0.01459053279807833`

## Self-Supervised Route-Free Evaluation

- `goal_success=true`
- `final_goal_distance=11.97125570872276`
- `min_goal_distance=11.97125570872276`
- `collision_count=0`
- `distance_traveled=81.83289448994688`
- `stuck_recovery_count=0`
- `reverse_count=0`
- `route_waypoint_count=0`
- `experience_route_point_count=0`

## Independent Route-Free Vs Route-Guided Baseline

- Output:
  `outputs/region_world_model_eval/johnson_valley_alt_mlp_generalization_20260707_live_baseline`
- Summary:
  `outputs/region_world_model_eval/johnson_valley_alt_mlp_generalization_20260707_live_baseline/region_world_model_evaluation_summary.json`
- Training record:
  `outputs/region_world_model_eval/johnson_valley_alt_mlp_generalization_20260707_live_baseline/training_run.json`
- Trajectory plot:
  `outputs/region_world_model_eval/johnson_valley_alt_mlp_generalization_20260707_live_baseline/region_world_model_trajectory.svg`

Command shape:

```powershell
python scripts\run_region_world_model_evaluation.py configs\tasks\beamng_johnson_valley_nav_alt.yaml --world-model-type mlp_dynamics --world-model outputs\region_self_supervised\johnson_valley_alt_mlp_generalization_20260706_live\model --eval-steps 1200 --evaluation-agent world_model_direct --evaluation-use-model-support-subgoals --evaluation-local-subgoal-distance-m 12 --include-route-guided-baseline
```

Route-free model-control result:

- `route_free_goal_success=true`
- `route_free_final_goal_distance=11.866867890459078`
- `route_free_min_goal_distance=11.866867890459078`
- `route_free_collision_count=0`
- `route_free_distance_traveled=81.95674861535258`
- `route_free_stuck_recovery_count=0`
- `route_free_reverse_count=0`
- `route_waypoint_count=0`
- `experience_route_point_count=0`

Route-guided comparison:

- `route_guided_goal_success=true`
- `route_guided_final_goal_distance=11.893627532222645`
- `route_guided_min_goal_distance=11.893627532222645`
- `route_guided_collision_count=0`
- `route_guided_distance_traveled=84.57453942873302`
- `route_guided_stuck_recovery_count=0`
- `route_guided_reverse_count=0`
- `route_waypoint_count=7`

## Interpretation

This fresh run validates the same-map alternate start/goal requirement with the
current workflow. The task is supplied as YAML data, the model is trained from
BeamNG route-aware collection, and the route-free evaluation reaches the goal
without code edits, collisions, reverse recovery, or BeamNG `ai_line` control.

The route-free controller still uses model-owned support subgoals learned from
the collected rollouts. That is different from pure direct goal pursuit, but it
does not inject the expert route into the evaluation task and remains the
current validated route-free path toward broader generalization.

## Remaining Risks

- This proves one alternate Johnson Valley task, not broad coverage of the map.
- The support-subgoal representation is still a local navigation prior, not a
  fully learned global world model.
- More random seeds, farther goals, and harsher terrain sections are still
  needed before calling route-free control generally robust.
