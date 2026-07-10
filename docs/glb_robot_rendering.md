# GLB Robot Rendering

The current datagen renderer composes human/NPC actors by merging Gaussian PLY
actors into the scene Gaussian model before rasterization. GLB robots are mesh
assets, so they use a separate foreground/background compositor:

1. Render the Gaussian Splatting scene normally with `render_label_paths.py`.
2. Save RGB frames, depth maps, and path-level camera metadata.
3. Render the robot GLB from the same per-frame camera.
4. Composite robot RGBA over the GS RGB frame.

The entry point is:

```bash
python scripts/convert_urdf_visuals_to_glb.py \
  --urdf data/g1_description/g1_29dof_mode_16.urdf \
  --output assets/robots/g1_29dof_mode_16.glb

python scripts/render_glb_robot_overlay.py \
  --camera-json <render_root>/<scene>/<label>_camera.json \
  --frames-dir <render_root>/<scene>/<label> \
  --robot-glb assets/robots/g1_29dof_mode_16.glb \
  --poses-json <robot_poses.json> \
  --output-dir <render_root>/<scene>/<label>_robot \
  --compose-mode foreground
```

If you already have an older `cuda121` environment, update it before running the
overlay renderer:

```bash
conda run -n cuda121 python -m pip install \
  pyrender==0.1.45 PyOpenGL==3.1.0 pyglet==1.5.31
```

`--compose-mode foreground` treats every visible GLB pixel as foreground. Use
`--compose-mode depth` to compare GLB depth against saved
`frame_XXXX_depth.png` maps and let the GS scene occlude the robot.

`data/g1_description` contains URDF/XML robot descriptions and STL meshes, not
a committed GLB. In this checkout, `data` is a symlink to the shared dataset
storage, so generated assets should be written to the repo, for example
`assets/robots/g1_29dof_mode_16.glb`. `scripts/convert_urdf_visuals_to_glb.py`
assembles the URDF visual meshes at zero joint position into a static GLB
suitable for the overlay renderer. Use `--joint-positions-json` if a different
nominal joint pose is needed for the static mesh export.

## Pose Contract

Robot motion is intentionally external to the visual compositor. IMO or another
planner/controller should emit a pose JSON with one entry per rendered frame:

```json
{
  "frames": [
    {"frame": 0, "position": [1.0, 2.0, 0.0], "yaw_deg": 45.0},
    {"frame": 1, "position": [1.1, 2.1, 0.0], "yaw_deg": 47.0}
  ]
}
```

Each pose can instead provide:

- `translation`, `xyz`, or scalar `x`/`y`/`z`
- `yaw_rad`, `yaw_deg`, or `quaternion_wxyz`
- `transform` or `matrix` as a full 4x4 world transform

The compositor can validate basic physical constraints before rendering:

```bash
--max-speed-mps 1.5 --max-yaw-rate-deg-s 120 --constraint-mode error
```

That validation is only a guardrail. The actual feasible robot trajectory
should come from IMO or the robot planner, especially when wheel limits,
collision envelopes, acceleration, or non-holonomic constraints matter.

## Coordinate Notes

NavDP scenes are Z-up. GLB/glTF assets are commonly Y-up, so the overlay script
defaults to `--glb-up-axis y` and maps the asset into the NavDP Z-up world before
applying the per-frame robot pose. If an asset is already authored as Z-up, pass
`--glb-up-axis z`.
