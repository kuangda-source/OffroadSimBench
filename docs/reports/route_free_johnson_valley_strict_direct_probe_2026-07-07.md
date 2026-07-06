# Johnson Valley Strict Direct Probe - 2026-07-07

## Goal

Quantify the remaining gap between validated model-support route-free control
and a stricter direct controller that receives only the region, start, goal, and
model dynamics checkpoint, without an experience corridor or ordered support
subgoals.

## Task And Model

- Task: `configs/tasks/beamng_johnson_valley_nav_test.yaml`
- Model:
  `outputs/region_self_supervised/johnson_valley_mlp_model_support_20260706_live/model`
- World model: `mlp_dynamics`
- Planner: `navigation_mpc`
- Evaluation agent: `world_model_direct`
- Evaluation steps: `1200`
- Route-free evaluation drive mode: `manual`
- Route-guided comparison drive mode: `manual`, using `route_world_model`

## Controller Fix Tested

The strict direct probe exposed that `_turn_arc_subgoal()` could be selected
before a valid direct lookahead point. At the Johnson Valley start, the previous
large-heading-error turn arc sent the local subgoal north of the vehicle, while
the final goal is southwest. The controller now prefers the direct lookahead
subgoal when that point remains inside the selected region; stuck recovery still
takes priority once the agent is already stalled.

Regression test:

- `tests/test_agents.py::test_world_model_direct_agent_prefers_direct_subgoal_when_it_stays_inside_region`

## Fresh BeamNG Runs

### Strict Direct Baseline Before Fix

- Output:
  `outputs/region_world_model_eval/johnson_valley_mlp_strict_direct_20260707_live`
- Trajectory plot:
  `outputs/region_world_model_eval/johnson_valley_mlp_strict_direct_20260707_live/region_world_model_trajectory.svg`

Route-free result:

- `route_free_goal_success=false`
- `route_free_min_goal_distance=113.11930280276982`
- `route_free_final_goal_distance=113.12295090727643`
- `route_free_collision_count=0`
- `route_free_distance_traveled=12.130666446812992`
- `route_free_stuck_recovery_count=1030`
- `route_free_reverse_count=0`

Route-guided comparison:

- `route_guided_goal_success=true`
- `route_guided_final_goal_distance=11.78346910770932`
- `route_guided_collision_count=0`

### Strict Direct After Direct-Lookahead Preference

- Output:
  `outputs/region_world_model_eval/johnson_valley_mlp_strict_direct_pref_direct_20260707_live`
- Trajectory plot:
  `outputs/region_world_model_eval/johnson_valley_mlp_strict_direct_pref_direct_20260707_live/region_world_model_trajectory.svg`

Route-free result:

- `route_free_goal_success=false`
- `route_free_min_goal_distance=112.7401054938678`
- `route_free_final_goal_distance=112.76178538245563`
- `route_free_collision_count=0`
- `route_free_distance_traveled=16.41454441841424`
- `route_free_stuck_recovery_count=1167`
- `route_free_reverse_count=0`

Route-guided comparison:

- `route_guided_goal_success=true`
- `route_guided_final_goal_distance=10.441313741618325`
- `route_guided_collision_count=0`

### Strict Direct With Reverse Last Resort

- Output:
  `outputs/region_world_model_eval/johnson_valley_mlp_strict_direct_reverse_last_20260707_live`
- Setting: `evaluation_allow_reverse_recovery=true`,
  `evaluation_reverse_recovery_after_steps=180`

Route-free result:

- `route_free_goal_success=false`
- `route_free_min_goal_distance=113.1716416195603`
- `route_free_final_goal_distance=113.36201969209954`
- `route_free_collision_count=0`
- `route_free_distance_traveled=28.62106600394507`
- `route_free_stuck_recovery_count=1128`
- `route_free_reverse_count=264`

Route-guided comparison:

- `route_guided_goal_success=true`
- `route_guided_final_goal_distance=10.506914088173211`
- `route_guided_collision_count=0`

### Unordered Support Field

- Output:
  `outputs/region_world_model_eval/johnson_valley_mlp_support_field_20260707_live`
- Setting: `evaluation_use_model_support_field_subgoals=true`,
  `evaluation_use_model_support_subgoals=false`

Route-free result:

- `route_free_goal_success=false`
- `route_free_min_goal_distance=115.31969884408291`
- `route_free_final_goal_distance=128.69526390313604`
- `route_free_collision_count=0`
- `route_free_distance_traveled=25.249171404079352`
- `route_free_stuck_recovery_count=1123`
- `route_free_reverse_count=0`

Route-guided comparison:

- `route_guided_goal_success=true`
- `route_guided_final_goal_distance=9.365479033035212`
- `route_guided_collision_count=0`

### Validated Model-Support Route Reference

- Output:
  `outputs/region_world_model_eval/johnson_valley_mlp_model_support_20260706_live_baseline`
- Setting: `evaluation_use_model_support_subgoals=true`

Route-free result:

- `route_free_goal_success=true`
- `route_free_min_goal_distance=11.939437166981389`
- `route_free_final_goal_distance=11.939437166981389`
- `route_free_collision_count=0`
- `route_free_distance_traveled=182.52625983648366`
- `route_free_stuck_recovery_count=0`
- `route_free_reverse_count=0`

## Diagnosis

The strict direct controller stalls near the start after only 12-16 m of travel.
The route-guided baseline succeeds on every run, so the map, vehicle, and route
are drivable. The direct controller fails because local geometry alone points it
into a difficult start-side terrain section; recovery actions then alternate
between lateral subgoals and high-throttle attempts without escaping.

The unordered support-field variant is not enough. It can see feasible points,
but without ordered or topological structure it may pick locally plausible
points that do not form a drivable sequence around the start-side obstacle. The
ordered model-support route remains the only current route-free configuration
that reaches the goal without collisions.

## Next Step

The next useful implementation target is not another throttle tweak. The model
adapter needs a route-free local traversability/topology memory that is richer
than unordered support points but less brittle than injecting an expert route at
evaluation time. Candidate implementation paths:

- learn or extract a sparse waypoint graph from collection rollouts;
- store edge validity and progress statistics instead of one ordered route;
- select local subgoals by graph/frontier search constrained to current region,
  with the final goal only as a directional bias;
- keep strict direct evaluation as the diagnostic lower bound, and keep
  model-support-route evaluation as the current demo-ready upper bound.
