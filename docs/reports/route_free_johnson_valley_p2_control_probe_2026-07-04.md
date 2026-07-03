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

## Stall-Memory Retest

Follow-up trace analysis found that BeamNG low-speed body jitter could clear
`_stuck_steps` before `_local_subgoal()` had a chance to keep using a recovery
target. The controller now treats movement below `0.12 m` per control step as
stall jitter, so direct control keeps its recovery state instead of repeatedly
returning to the same straight-line local subgoal.

Strict direct without reverse still does not escape the first local trap:

- Output: `outputs/region_world_model_eval/p2_stall_memory_strict_direct_420_20260704/region_world_model_evaluation_summary.json`.
- Minimum goal distance: `110.958 m`.
- Final goal distance: `110.958 m`.
- Distance traveled: `12.317 m`.
- Stuck recovery count: `376`.
- Reverse count: `0`.

With reverse enabled as a last resort, strict direct now escapes the first trap
but stalls on a second local platform:

- 420-step output:
  `outputs/region_world_model_eval/p2_stall_memory_reverse_direct_420_20260704/region_world_model_evaluation_summary.json`.
- 420-step minimum/final goal distance: `98.373 m` / `98.373 m`.
- 1200-step output:
  `outputs/region_world_model_eval/p2_stall_memory_reverse_direct_1200_20260704/region_world_model_evaluation_summary.json`.
- 1200-step minimum/final goal distance: `98.203 m` / `98.272 m`.
- 1200-step collision count: `0`.
- 1200-step distance traveled: `34.139 m`.
- 1200-step stuck recovery count: `1095`.
- 1200-step reverse count: `302`.

Conclusion: persistent stall memory plus last-resort reverse improves the strict
direct baseline from the old `~108-110 m` plateau to `~98 m`, but it is still far
from the `<50 m` first-stage target. The remaining blocker is not a transient
stuck counter reset; direct control still lacks a traversability-aware way to
choose a better local direction after escaping the first obstacle cluster.

Follow-up negative experiment: treating every low-throttle/braking/no-progress
state as stuck made the BeamNG closed loop worse. It triggered recovery too
early during a slow but still useful turn and regressed the 420-step strict
direct probe:

- Broad conservative-stuck output:
  `outputs/region_world_model_eval/p2_conservative_stuck_physical_direct_420_20260704/region_world_model_evaluation_summary.json`.
- Minimum/final goal distance: `104.437 m` / `104.437 m`.
- Collision count: `0`.
- Distance traveled: `27.222 m`.
- Stuck recovery count: `342`.
- Reverse count: `74`.

This rule was not kept in code. The next productive direction is a
traversability-aware local direction prior, not a broader stuck trigger.

## Support-Risk Data-Flow Retest

Another trace issue was in `TinyLearnedWorldModel` support risk. The model used
the minimum distance from any predicted state to any support point, so a
trajectory could report `support_risk=0` just because its first predicted state
was still near known data, even if later predicted states left the known
drivable support. This prevented the planner risk term from expressing
"trajectory leaves known feasible data."

Using the maximum predicted support distance made the risk signal too
conservative and regressed the strict direct 420-step probe:

- Max-distance output:
  `outputs/region_world_model_eval/p2_support_risk_worst_direct_420_20260704/region_world_model_evaluation_summary.json`.
- Minimum/final goal distance: `104.436 m` / `104.438 m`.
- Final planner `risk_cost`: `0.420`.

The retained implementation uses the mean nearest-support distance over the
predicted trajectory. This keeps nonzero risk for trajectories that leave known
support while avoiding one-step over-penalization.

- Mean-risk 420-step output:
  `outputs/region_world_model_eval/p2_support_risk_mean_direct_420_20260704/region_world_model_evaluation_summary.json`.
- Mean-risk 420-step minimum/final goal distance: `98.292 m` / `98.314 m`.
- Mean-risk 1200-step output:
  `outputs/region_world_model_eval/p2_support_risk_mean_direct_1200_20260704/region_world_model_evaluation_summary.json`.
- Mean-risk 1200-step minimum/final goal distance: `98.138 m` / `98.233 m`.
- Mean-risk 1200-step collision count: `0`.
- Mean-risk 1200-step distance traveled: `35.234 m`.
- Mean-risk 1200-step stuck recovery count: `1092`.
- Mean-risk 1200-step reverse count: `302`.

The same mean-risk change preserves the model-owned support-route demo:

- Support-route output:
  `outputs/region_world_model_eval/p2_support_risk_mean_support_route_1200_20260704/region_world_model_evaluation_summary.json`.
- Goal success: `true`.
- Final/minimum goal distance: `11.071 m` / `11.071 m`.
- Collision count: `0`.
- Stuck recovery count: `0`.
- Reverse count: `0`.

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
