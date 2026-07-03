# Johnson Valley Route-Free P2 Control Probe - 2026-07-04

## Scope

Task: `configs/tasks/beamng_johnson_valley_nav_test.yaml`

Goal: improve `world_model_direct` control after P1 route-aware collection, while keeping strict route-free direct evaluation separate from any experience-corridor aided evaluation.

## Strict Direct Status

The strict direct route-free baseline still fails:

- Inputs: region, start, goal only.
- Expert route: removed.
- Experience corridor: disabled.
- Reference output: `outputs/region_world_model_eval/p1_existing_demo_ready_1200_20260703_071234/region_world_model_evaluation_summary.json`.
- Minimum goal distance: `109.825 m`.
- Final goal distance: `109.835 m`.
- Collision count: `0`.
- Stuck recovery count: `1016`.
- Reverse count: `0`.

P2 controller changes reduced some local failure modes, but strict direct is not yet below the `<50 m` acceptance target.

## Controller Patch Probes

The direct controller now:

- tracks progress against the current local subgoal instead of only the final goal;
- detects physical stalls when the subgoal changes;
- delays reverse recovery and uses low-speed forward recovery first;
- generates turn-arc and lateral escape subgoals inside the selected region;
- exposes evaluation controls through `RegionWorldModelEvaluationRequest`.

Representative 420-step probe with last-resort reverse enabled:

- Output: `outputs/region_world_model_eval/p2_reverse_last_resort_420_20260703_143126/region_world_model_evaluation_summary.json`.
- Minimum goal distance: `108.471 m`.
- Final goal distance: `108.636 m`.
- Collision count: `0`.
- Distance traveled: `15.401 m`.
- Stuck recovery count: `194`.
- Reverse count: `12`.

Conclusion: reverse recovery alone does not solve route-free navigation and can add unhelpful maneuvering.

## Experience-Corridor Evaluation

Standalone model evaluation can now explicitly rebuild an experience corridor from the model training episode metadata. This corridor comes from collected BeamNG episode traces, not from the expert route injected into the evaluation scenario.

420-step probe:

- Output: `outputs/region_world_model_eval/p2_model_experience_corridor_420_20260704_012932/region_world_model_evaluation_summary.json`.
- Experience route points: `46`.
- Minimum goal distance: `91.082 m`.
- Final goal distance: `91.082 m`.
- Collision count: `0`.
- Distance traveled: `63.363 m`.
- Stuck recovery count: `0`.
- Reverse count: `0`.

1200-step probe:

- Output: `outputs/region_world_model_eval/p2_model_experience_corridor_1200_20260704_013100/region_world_model_evaluation_summary.json`.
- Experience route points: `46`.
- Goal success: `true`.
- Minimum goal distance: `11.877 m`.
- Final goal distance: `11.877 m`.
- Goal radius: `12.0 m`.
- Collision count: `0`.
- Distance traveled: `175.708 m`.
- Stuck recovery count: `0`.
- Reverse count: `0`.

This satisfies the current Johnson Valley task as an intermediate route-free-with-experience-corridor demo. It does not satisfy the stricter direct-only P5 target yet.

## GUI Impact

The GUI can now:

- list experience-corridor-validated tiny/direct configs as demo-ready when validation proves route-free, model-controlled, no-collision goal success with zero route waypoints;
- pass `use_experience_corridor=True` into direct model evaluation when the selected world-model config is marked with `experience_corridor`;
- keep strict direct evaluation as the service default, so P0 baselines remain clean.

Local demo config registered for this machine:

`johnson_valley_tiny_experience_corridor_20260704`

## Next Work

- Keep strict direct P5 gate active: route-free direct must reach `<50 m` without experience corridor.
- Add a planner cost term for distance from the learned experience corridor instead of only using it for local subgoal selection.
- Continue improving strict local navigation with better traversability and stall recovery before adding a second model.
