# Scene Placement and Orientation Reference

Use this document when adding humans, robots, NPCs, meshes, or other foreground
objects to rendered NavDP Gaussian Splatting scenes. The goal is to keep every
renderer and overlay script in the same world frame.

## Canonical Scene Frame

NavDP scene/world coordinates are Z-up:

- `+Z` is vertical up.
- `X/Y` are the scene floor-plane coordinates.
- The floor plane for a scene is `occupancy.lower[2]`, also exposed as
  `meta["lower_z"]` in render paths.
- A placed object's `position[2]` should usually be
  `floor_z + foot_offset`, where `foot_offset` is only for intentional visual
  clearance or model-specific correction.

Do not place an object by using a mesh center as the ground point. The renderer
expects the pose translation to be the object's ground-contact origin after
local asset normalization.

## Pathplanner to TeleSim Coordinates

MassGen scenario files store robot and human trajectories as Pathplanner
`map_pose` values. For the CHINGMU Pathplanner scenarios used by the MassGen
rollouts, those `map_pose.x` / `map_pose.y` values already align with the
TeleSim scene BEV and occupancy metadata. The MassGen render bridge must not
mirror XY about the occupancy center before rendering; doing so moves cameras
and humans to the opposite side of the map.

The exact pipeline for generated camera label paths is:

```text
Pathplanner scenario:
  robots[].trajectory[].map_pose.{x,y,yaw}

utils.massgen_render_manifest:
  camera.trajectory[].position = [map_pose.x, map_pose.y, 0.0]
  camera.trajectory[].yaw_rad = map_pose.yaw

navdp_datagen.massgen.render_executor label path:
  raster_world.{x,y,z} = camera.trajectory[].position
  raster_pixel.u = round((raster_world.x - occupancy.left) / occupancy.scale)
  raster_pixel.v = round((occupancy.top - raster_world.y) / occupancy.scale)
  metadata.coordinate_transform = identity_xy

render_label_paths_telesim:
  load raster_world and raster_pixel
  derive affine transform from raster_world to raster_pixel
  render with --no-mirror-translation
```

The exact pipeline for generated human actor placement is:

```text
Pathplanner scenario:
  humans[].trajectory[].map_pose.{x,y,yaw}

utils.massgen_render_manifest:
  actor trajectory position = [map_pose.x, map_pose.y, 0.0]
  actor trajectory yaw_rad = map_pose.yaw

navdp_datagen.massgen.render_executor actor plan:
  actor frame position.{x,y,z} = actor trajectory position
  actor sample yaw = atan2(cos(pathplanner_yaw), sin(pathplanner_yaw))
  actor frame yaw_rad = actor sample yaw + pi

render_label_paths_telesim:
  apply actor frame position directly as world translation
  apply actor frame yaw_rad + actor plan yaw_offset_rad
  use floor z when actor z_mode is "floor"
```

Generated label JSON records this under
`metadata.coordinate_pipeline`. Generated actor bundle JSON records the same
contract under top-level `coordinate_pipeline`.

MassGen actor plans set `yaw_offset_rad` to `0.0`; the `+pi` is applied before
the bundle frame is written so stationary actors use the same canonical human
asset convention as moving actors (`atan2(direction_x, direction_y) + pi`). Do
not pre-negate Pathplanner yaw before this step: that double-compensates the
asset convention and flips queue/group humans outward by 180 degrees.

## Asset Normalization

Every foreground asset should be converted into a local frame where local
`min_z == 0` before the world pose is applied.

Human Gaussian actors follow this convention in `utils/telesim_actor_utils.py`:

1. Apply `ACTOR_AXIS_ALIGNMENT_MATRIX`.
2. Scale the actor to the requested height.
3. Translate every animation frame by `-global_min_z`.
4. Place the actor at `floor_z + actor_foot_offset`.

GLB robot rendering follows the same convention in
`utils/glb_robot_compositor.py`:

1. Apply the asset up-axis alignment (`--glb-up-axis`).
2. Optionally scale to `--target-height`.
3. Translate by `-aligned_min_z`.
4. Apply the per-frame world pose.

For the generated G1 URDF GLB at
`assets/robots/g1_29dof_mode_16.glb`, use `--glb-up-axis z`. The URDF assembly
already produces a Z-up robot: ankle links are at the local minimum Z and the
head is at the local maximum Z.

Quick bounds check:

```bash
python - <<'PY'
import trimesh
scene = trimesh.load("assets/robots/g1_29dof_mode_16.glb", force="scene")
print(scene.bounds)
print("extents:", scene.bounds[1] - scene.bounds[0])
PY
```

If feet/head appear reversed in the bounds, fix the asset export or
`--glb-up-axis` before debugging yaw.

## World Pose Contract

Foreground pose JSON should describe object-to-world transforms in NavDP world
coordinates. For simple planar motion:

```json
{
  "frames": [
    {"frame": 0, "position": [1.0, 2.0, 0.1], "yaw_rad": 0.0}
  ]
}
```

`position` is the object ground-contact origin in world coordinates, not the
visual mesh center. `yaw_rad` is a rotation about world `+Z`.

The GLB compositor builds:

```text
object_to_world = translation(position) @ yaw_about_world_z @ asset_normalizer
```

If a controller provides a full 4x4 `transform`, it must already be an
object-to-world transform in the same NavDP world frame.

## Yaw and Forward Axis

Yaw depends on the source asset's local forward direction. For a path direction
`d = (dx, dy)`:

- asset `+X` forward: `yaw = atan2(dy, dx)`
- asset `-X` forward: `yaw = atan2(-dy, -dx)`
- asset `+Y` forward: `yaw = atan2(-dx, dy)`
- asset `-Y` forward: `yaw = atan2(dx, -dy)`

The G1 example runner exposes this as `--robot-forward-axis`; current G1 URDF
usage defaults to `--robot-forward-axis x`. If the robot is upright and grounded
but faces sideways/backward, change `--robot-forward-axis` or add
`--robot-yaw-offset-deg`. Do not fix forward-facing errors by changing
`--glb-up-axis`.

Human actors use a different source asset convention. The TeleSim human path
uses:

```python
theta = math.atan2(direction_xy[0], direction_xy[1]) + math.pi
```

That formula is correct for the aligned human PLY actor source, but it should
not be copied blindly to GLB/URDF assets.

Stationary MassGen human actors carry conventional Pathplanner yaw in
`map_pose.yaw`, where the intended map-forward vector is
`[cos(yaw), sin(yaw)]`. The MassGen render executor stores the corresponding
human actor sample yaw internally as:

```python
sample_yaw = math.atan2(math.cos(pathplanner_yaw), math.sin(pathplanner_yaw))
```

It then writes actor-plan frame yaw as `sample_yaw + math.pi`; TeleSim applies
that frame yaw plus the plan's `yaw_offset_rad`, which MassGen sets to `0.0`.
Negating the planner forward vector before this step flips queue and group
actors outward by 180 degrees.

## Follow-Camera Placement

Human follow-camera rendering keeps the actor in front of the camera by
separating camera and actor distances along the same path:

```text
max_camera_distance = max(total_path_length - follow_distance, 0)
camera_distance = min(path_distance, max_camera_distance)
actor_distance = min(camera_distance + follow_distance, total_path_length)
```

When rendering a base GS sequence without an in-render actor, use
`render_label_paths.py --limit-camera-to-follow` if a foreground overlay actor
or robot will be added later. Otherwise the camera can advance all the way to
the goal while the overlay object remains at the goal, making the object appear
too close, clipped, or floating.

`scripts/run_g1_robot_follow_example.py` passes `--limit-camera-to-follow` for
this reason.

## Camera Metadata to Pyrender

`render_label_paths.py` stores camera matrices in the same transposed form used
by the CUDA GS renderer. Do not feed `camera_to_world` directly to pyrender.

For pyrender/OpenGL mesh overlays:

1. Transpose the stored `camera_to_world`.
2. Flip camera Y and Z axes.
3. Confirm the rotation determinant is `+1`.

The implemented conversion is:

```python
pose = np.asarray(frame["camera_to_world"]).T.copy()
pose[:3, 1] *= -1.0
pose[:3, 2] *= -1.0
```

Sanity check:

```bash
python - <<'PY'
import json
import numpy as np
from utils.glb_robot_compositor import camera_metadata_to_pyrender_pose

frame = json.load(open("data2/g1_robot_follow_capcheck/base_gs/0001_839920/49_camera.json"))["frames"][0]
pose = camera_metadata_to_pyrender_pose(frame)
print("det:", np.linalg.det(pose[:3, :3]))
print("translation:", pose[:3, 3])
PY
```

The determinant should be close to `1.0`. A determinant near `-1.0` means the
camera basis is reflected; GLB objects may appear upside down, mirrored, or
rotated inconsistently as yaw changes.

## Debug Checklist

Use this order when an object appears misplaced:

1. **Wrong height or floating:** check `floor_z`, `foot_offset`, and local
   `min_z == 0` after asset normalization.
2. **Upside down or ceiling-side:** check `--glb-up-axis`, then check the
   camera-to-pyrender conversion determinant.
3. **Facing wrong direction:** check local forward axis and yaw offset.
4. **Object clipped or too close:** check `--limit-camera-to-follow` and
   camera/person distance in `<label>_follow_path.json`.
5. **Object disappears in some frames:** check that pose frame IDs match camera
   frame IDs and that the object is in front of the camera frustum.
6. **Depth occlusion looks wrong:** first test `--compose-mode foreground`;
   then debug depth maps and `--depth-bit-depth`.

Useful per-frame distance check:

```bash
python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path("data2/g1_robot_follow_example/base_gs/0001_839920")
frames = json.load(open(root / "49_follow_path.json"))["frames"]
for i in [0, len(frames)//2, len(frames)-1]:
    cam = np.array(frames[i]["camera_world"], dtype=float)
    obj = np.array(frames[i]["person_world"], dtype=float)
    print(i, np.linalg.norm(obj - cam), cam.tolist(), obj.tolist())
PY
```

The distance should remain close to the requested follow distance unless the
path is shorter than that distance.

## Code References

- Human actor normalization:
  `utils/telesim_actor_utils.py::load_actor_sequence`
- Human actor placement:
  `render_label_paths_telesim.py::build_actor_follow_plans`
- Base camera capping:
  `render_label_paths.py --limit-camera-to-follow`
- GLB pose parsing and camera conversion:
  `utils/glb_robot_compositor.py`
- G1 example pose generation:
  `scripts/run_g1_robot_follow_example.py`
