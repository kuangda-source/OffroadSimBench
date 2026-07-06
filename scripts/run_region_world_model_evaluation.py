"""Run route-free region world-model evaluation with optional baseline comparison."""

from __future__ import annotations

import argparse
import json

from desktop_app.services import RegionWorldModelEvaluationRequest, run_region_world_model_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_path", help="navigation_region_v1 task YAML.")
    parser.add_argument("--world-model-type", default="tiny_learned", choices=["tiny_learned", "mlp_dynamics", "le_wm", "simple_kinematic"])
    parser.add_argument("--world-model", required=True, help="World model checkpoint or model directory.")
    parser.add_argument("--output-dir", default="", help="Output directory for evaluation summary.")
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planner", default="navigation_mpc")
    parser.add_argument("--planner-horizon", type=int, default=6)
    parser.add_argument("--planner-samples", type=int, default=32)
    parser.add_argument("--planner-iterations", type=int, default=3)
    parser.add_argument("--planner-goal-weight", type=float, default=None)
    parser.add_argument("--planner-progress-weight", type=float, default=None)
    parser.add_argument("--planner-risk-weight", type=float, default=None)
    parser.add_argument("--planner-heading-weight", type=float, default=None)
    parser.add_argument("--evaluation-agent", default="world_model_direct")
    parser.add_argument("--evaluation-allow-reverse-recovery", action="store_true")
    parser.add_argument("--evaluation-reverse-recovery-after-steps", type=int, default=96)
    parser.add_argument("--evaluation-local-subgoal-distance-m", type=float, default=12.0)
    parser.add_argument("--evaluation-use-model-support-subgoals", action="store_true")
    parser.add_argument("--evaluation-use-model-support-field-subgoals", action="store_true")
    parser.add_argument("--evaluation-use-model-support-graph-subgoals", action="store_true")
    parser.add_argument("--use-experience-corridor", action="store_true")
    parser.add_argument("--experience-route-min-spacing-m", type=float, default=4.0)
    parser.add_argument("--experience-route-max-points", type=int, default=120)
    parser.add_argument("--include-route-guided-baseline", action="store_true")
    parser.add_argument("--beamng-gfx", default="vk")
    parser.add_argument("--close-beamng", action="store_true")
    parser.add_argument("--step-delay-sec", type=float, default=0.0)
    parser.add_argument("--post-run-hold-sec", type=float, default=0.0)
    args = parser.parse_args()

    payload = run_region_world_model_evaluation(
        RegionWorldModelEvaluationRequest(
            task_path=args.task_path,
            world_model_type=args.world_model_type,
            world_model_path=args.world_model,
            output_dir=args.output_dir,
            eval_steps=args.eval_steps,
            seed=args.seed,
            planner=args.planner,
            planner_horizon=args.planner_horizon,
            planner_samples=args.planner_samples,
            planner_iterations=args.planner_iterations,
            planner_goal_weight=args.planner_goal_weight,
            planner_progress_weight=args.planner_progress_weight,
            planner_risk_weight=args.planner_risk_weight,
            planner_heading_weight=args.planner_heading_weight,
            evaluation_agent=args.evaluation_agent,
            evaluation_allow_reverse_recovery=args.evaluation_allow_reverse_recovery,
            evaluation_reverse_recovery_after_steps=args.evaluation_reverse_recovery_after_steps,
            evaluation_local_subgoal_distance_m=args.evaluation_local_subgoal_distance_m,
            evaluation_use_model_support_subgoals=args.evaluation_use_model_support_subgoals,
            evaluation_use_model_support_field_subgoals=args.evaluation_use_model_support_field_subgoals,
            evaluation_use_model_support_graph_subgoals=args.evaluation_use_model_support_graph_subgoals,
            use_experience_corridor=args.use_experience_corridor,
            experience_route_min_spacing_m=args.experience_route_min_spacing_m,
            experience_route_max_points=args.experience_route_max_points,
            include_route_guided_baseline=args.include_route_guided_baseline,
            beamng_gfx=args.beamng_gfx,
            close_beamng=args.close_beamng,
            step_delay_sec=args.step_delay_sec,
            post_run_hold_sec=args.post_run_hold_sec,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
