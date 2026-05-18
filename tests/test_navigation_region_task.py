from __future__ import annotations

from pathlib import Path

from offroad_sim.tasks import NavigationRegionTask, load_navigation_region_task


def test_navigation_region_task_builds_collection_and_eval_scenarios(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        """
task_id: region_demo
task_type: navigation_region_v1
map_id: gridmap_v2_region_001
backend_targets: [beamng]
level: gridmap_v2
region:
  polygon:
    - [0.0, -160.0]
    - [30.0, -160.0]
    - [30.0, -260.0]
    - [0.0, -260.0]
start_pose:
  pos: [1.0, -170.0, 100.6]
  yaw: -1.57
goal:
  pos: [4.0, -240.0]
  radius: 6.0
expert_route:
  - [1.0, -170.0]
  - [2.0, -205.0]
  - [4.0, -240.0]
constraints:
  max_steps: 300
  max_collision_count: 0
beamng:
  vehicle_model: pickup
  ai_line_speed: 11.0
""".strip(),
        encoding="utf-8",
    )

    task = load_navigation_region_task(path)
    collection = task.to_beamng_scenario(mode="collection")
    evaluation = task.to_beamng_scenario(mode="evaluation")

    assert task.task_id == "region_demo"
    assert task.contains_point((10.0, -200.0)) is True
    assert task.contains_point((40.0, -200.0)) is False
    assert collection["metadata"]["beamng"]["route"] == [[1.0, -170.0], [2.0, -205.0], [4.0, -240.0]]
    assert "route" not in evaluation["metadata"]["beamng"]
    assert evaluation["task"]["start"] == [1.0, -170.0]
    assert evaluation["task"]["goal"] == [4.0, -240.0]


def test_navigation_region_task_can_use_agent_control_for_evaluation() -> None:
    task = NavigationRegionTask(
        task_id="manual_eval",
        map_id="gridmap_v2_region_001",
        level="gridmap_v2",
        region_polygon=[(0.0, -160.0), (30.0, -160.0), (30.0, -230.0), (0.0, -230.0)],
        start_pos=(1.0, -170.0, 100.6),
        start_yaw=-1.57,
        goal_pos=(6.0, -215.0),
        goal_radius=6.0,
        expert_route=[(1.0, -170.0), (4.0, -195.0), (6.0, -215.0)],
        beamng={"drive_mode": "ai_line", "evaluation_drive_mode": "manual"},
    )

    collection = task.to_beamng_scenario(mode="collection")
    evaluation = task.to_beamng_scenario(mode="evaluation")

    assert collection["metadata"]["beamng"]["drive_mode"] == "ai_line"
    assert evaluation["metadata"]["beamng"]["drive_mode"] == "manual"
    assert "route" not in evaluation["metadata"]["beamng"]


def test_navigation_region_task_rejects_missing_expert_route_for_collection() -> None:
    task = NavigationRegionTask(
        task_id="no_expert",
        map_id="gridmap_v2_region_001",
        level="gridmap_v2",
        region_polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        start_pos=(1.0, 1.0, 0.5),
        start_yaw=0.0,
        goal_pos=(8.0, 8.0),
        goal_radius=2.0,
        expert_route=[],
    )

    try:
        task.to_beamng_scenario(mode="collection")
    except ValueError as exc:
        assert "expert_route" in str(exc)
    else:
        raise AssertionError("Expected expert_route validation.")
