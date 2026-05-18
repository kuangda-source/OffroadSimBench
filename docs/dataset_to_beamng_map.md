# Dataset To BeamNG Map Notes

## Why Direct Conversion Is Hard

Real off-road datasets such as ORFD are usually image-centric: RGB, labels,
depth-like assets, or sparse lidar frames. A BeamNG map needs simulator-ready
geometry, scale, collision, materials, spawn points, AI/navigation metadata, and
roads or drivable terrain that the physics engine can use. The missing pieces
are the difficult part.

Key gaps:

- Metric scale and global pose are often incomplete or sequence-local.
- Monocular RGB/segmentation alone cannot recover reliable terrain height.
- Sparse depth/lidar creates holes, occlusions, and noisy elevation estimates.
- BeamNG needs collision-ready terrain or mesh geometry, not just images.
- Decal roads are visual overlays; they do not by themselves provide collision.
- Off-road datasets often contain vegetation, mud, rocks, and traversability
  cues that are semantic rather than geometric.
- BeamNG map packaging requires materials, level metadata, object placement,
  spawn points, and validation in the editor/runtime.

## Practical Path

Use three layers instead of trying to reconstruct a full photoreal map in one
step.

1. **Planner cost map first**
   Convert labels/depth/lidar into a local traversability grid. Use it for
   world-model/planner cost, boundary penalties, and start-goal experiments.
   This is the fastest path to meaningful algorithm evaluation.

2. **Heightmap draft second**
   Export a 16-bit grayscale heightmap plus optional road/route JSON. BeamNG's
   Terrain and Road Importer can use this style of asset as a World Editor
   starting point. This is suitable for local terrain shape and route demos.

3. **Packaged map later**
   Turn the draft into a real BeamNG level: terrain blocks, materials, static
   objects, collision meshes, roads, decals, spawn points, and scenario files.
   This stage needs editor validation and should be treated as map authoring,
   not pure dataset conversion.

## Current Project State

OffroadSimBench now supports the first layer:

- GUI task editor for region polygon, start, goal, and expert route.
- `navigation_region_v1` YAML task files.
- BeamNG observations expose `navigation_region`.
- `world_model_cem` penalizes trajectories outside the selected polygon.
- The local LE-WM-compatible cost model receives `region_polygon` and penalizes
  out-of-region rollouts.

The project also has a terrain draft exporter that writes a 16-bit heightmap,
OBJ mesh preview, and manifest from ORFD-like assets. That output is not a
packaged BeamNG level yet.

## Recommended Next Implementation

Add a dedicated `beamng_map_draft_v1` exporter:

- Input: ORFD sequence, selected frame/range, optional manual region task.
- Output: 16-bit heightmap PNG, route/road JSON, traversability mask, preview,
  and a manifest describing scale, origin, and coordinate transform.
- Validation: render preview, verify finite height values, verify start/goal are
  inside traversability bounds, and run a BeamNG import smoke only when the user
  explicitly launches BeamNG.

This gives a controlled bridge from dataset assets to BeamNG authoring without
pretending that a single RGB dataset can automatically become a complete
physical world.
