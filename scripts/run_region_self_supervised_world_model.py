"""Run route-free region self-supervised world-model training and evaluation."""

from __future__ import annotations

import argparse
import json

from desktop_app.services import RegionSelfSupervisedWorldModelRequest, run_region_self_supervised_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_path", help="navigation_region_v1 task YAML.")
    parser.add_argument("--output-dir", default="", help="Output directory for model and summary.")
    parser.add_argument("--world-model-type", default="tiny_learned", choices=["tiny_learned", "mlp_dynamics"])
    parser.add_argument("--collect-steps", type=int, default=1000)
    parser.add_argument("--collect-rollouts", type=int, default=1)
    parser.add_argument("--min-collection-goal-progress-ratio", type=float, default=0.0)
    parser.add_argument("--collection-goal-bias-interval", type=int, default=1)
    parser.add_argument("--collection-goal-corridor-interval", type=int, default=1)
    parser.add_argument("--collection-goal-corridor-lateral-m", type=float, default=2.0)
    parser.add_argument("--collection-coverage-grid-size", type=int, default=0)
    parser.add_argument("--collection-coverage-target-interval", type=int, default=0)
    parser.add_argument("--collection-max-target-steps", type=int, default=80)
    parser.add_argument("--collection-strategy", default="region_explorer", choices=["region_explorer", "route_aware"])
    parser.add_argument("--collection-route-target-interval", type=int, default=0)
    parser.add_argument("--collection-route-lateral-m", type=float, default=0.0)
    parser.add_argument("--collection-multi-start", action="store_true")
    parser.add_argument("--collection-multi-start-lateral-m", type=float, default=0.0)
    parser.add_argument("--min-route-coverage-ratio", type=float, default=0.0)
    parser.add_argument("--min-goal-zone-coverage", type=float, default=0.0)
    parser.add_argument("--max-collection-min-goal-distance-m", type=float, default=0.0)
    parser.add_argument("--min-unique-region-cells", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planner", default="navigation_mpc")
    parser.add_argument("--planner-horizon", type=int, default=6)
    parser.add_argument("--planner-samples", type=int, default=32)
    parser.add_argument("--planner-iterations", type=int, default=3)
    parser.add_argument("--evaluation-agent", default="world_model_direct")
    parser.add_argument("--evaluation-route-mode", default="route_free", choices=["route_free", "task_route"])
    parser.add_argument("--no-experience-corridor", action="store_true")
    parser.add_argument("--evaluation-local-subgoal-distance-m", type=float, default=12.0)
    parser.add_argument("--evaluation-use-model-support-subgoals", action="store_true")
    parser.add_argument("--evaluation-use-model-support-field-subgoals", action="store_true")
    parser.add_argument("--evaluation-use-model-support-graph-subgoals", action="store_true")
    parser.add_argument("--beamng-gfx", default="vk")
    parser.add_argument("--close-beamng", action="store_true")
    parser.add_argument("--step-delay-sec", type=float, default=0.02)
    parser.add_argument("--post-run-hold-sec", type=float, default=20.0)
    parser.add_argument("--register-world-model-config", action="store_true")
    parser.add_argument("--world-model-config-path", default="")
    args = parser.parse_args()

    payload = run_region_self_supervised_world_model(
        RegionSelfSupervisedWorldModelRequest(
            task_path=args.task_path,
            world_model_type=args.world_model_type,
            output_dir=args.output_dir,
            collect_steps=args.collect_steps,
            collect_rollouts=args.collect_rollouts,
            min_collection_goal_progress_ratio=args.min_collection_goal_progress_ratio,
            collection_goal_bias_interval=args.collection_goal_bias_interval,
            collection_goal_corridor_interval=args.collection_goal_corridor_interval,
            collection_goal_corridor_lateral_m=args.collection_goal_corridor_lateral_m,
            collection_coverage_grid_size=args.collection_coverage_grid_size,
            collection_coverage_target_interval=args.collection_coverage_target_interval,
            collection_max_target_steps=args.collection_max_target_steps,
            collection_strategy=args.collection_strategy,
            collection_route_target_interval=args.collection_route_target_interval,
            collection_route_lateral_m=args.collection_route_lateral_m,
            collection_multi_start=args.collection_multi_start,
            collection_multi_start_lateral_m=args.collection_multi_start_lateral_m,
            min_route_coverage_ratio=args.min_route_coverage_ratio,
            min_goal_zone_coverage=args.min_goal_zone_coverage,
            max_collection_min_goal_distance_m=args.max_collection_min_goal_distance_m,
            min_unique_region_cells=args.min_unique_region_cells,
            eval_steps=args.eval_steps,
            seed=args.seed,
            planner=args.planner,
            planner_horizon=args.planner_horizon,
            planner_samples=args.planner_samples,
            planner_iterations=args.planner_iterations,
            evaluation_agent=args.evaluation_agent,
            evaluation_route_mode=args.evaluation_route_mode,
            use_experience_corridor=not args.no_experience_corridor,
            evaluation_local_subgoal_distance_m=args.evaluation_local_subgoal_distance_m,
            evaluation_use_model_support_subgoals=args.evaluation_use_model_support_subgoals,
            evaluation_use_model_support_field_subgoals=args.evaluation_use_model_support_field_subgoals,
            evaluation_use_model_support_graph_subgoals=args.evaluation_use_model_support_graph_subgoals,
            beamng_gfx=args.beamng_gfx,
            close_beamng=args.close_beamng,
            step_delay_sec=args.step_delay_sec,
            post_run_hold_sec=args.post_run_hold_sec,
            register_world_model_config=args.register_world_model_config,
            world_model_config_path=args.world_model_config_path,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
