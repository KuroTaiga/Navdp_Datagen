# Camera and Sensor Defaults

Date: 2026-08-05
Branch: `massgen`

This document records the fallback camera profiles to use until a render job can
import exact robot sensor settings from an Isaac Sim/OpenUSD rig. Imported rigs
always take priority over these defaults.

## Default Policy

Use defaults in this order:

1. If a job provides an Isaac Sim/OpenUSD robot rig with camera prims, use the
   authored sensor transforms, optical attributes, resolution, clipping range,
   rate, and modality settings.
2. If the job asks for a G1 robot but no sensor rig is available, use the
   provisional `g1_head_fpv_default` profile below and emit a warning that the
   mount transform is a local renderer default.
3. If the job is an existing NavDP FPV/follow render with no physical robot
   sensor rig, use `navdp_legacy_fpv` for dataset compatibility.
4. Use raw OpenUSD fallback camera attributes only when a USD camera prim exists
   but does not author its optics. That fallback is much narrower than our
   legacy training camera and should not silently become the MassGen default.

All fallback camera intrinsics are pinhole, square-pixel, centered principal
point, and no lens distortion unless an imported rig explicitly provides a
distortion model.

## Recommended Profiles

| Profile | Intended Use | Resolution | Intrinsics / FOV | Clipping | Height / Mount | Orientation |
| --- | --- | --- | --- | --- | --- | --- |
| `navdp_legacy_fpv` | Backwards-compatible FPV/follow rendering | `960x720` | `fov_y=70 deg`, `fov_x=86.067 deg`, `fx=fy=514.133`, `cx=480`, `cy=360` | `znear=0.001 m`, `zfar=30 m` | `camera_z = occupancy_upper_z + 0.3 m`; legacy CHINGMU assumption is about `1.3 m` when `upper_z=1.0 m` | Look along path with `look_ahead=2.0 m`, `look_down=0.1 m`; about `2.862 deg` downward pitch on straight paths |
| `g1_head_fpv_default` | Provisional G1 robot-mounted RGB/depth view when no USD rig exists | `960x720` | Same as `navdp_legacy_fpv` for training-data continuity | `znear=0.05 m`, `zfar=30 m` for a physical robot sensor; use `0.001 m` only for legacy compatibility runs | Robot-relative `translation_m=[0.10, 0.0, 1.20]` in `+X forward, +Y left, +Z up`; robot visibility height remains `1.3 m` | `yaw=0 deg`, `pitch=-5 deg`, `roll=0 deg`; follows robot yaw |
| `openusd_camera_fallback` | USD camera prim with missing authored optics | Render product setting | OpenUSD defaults imply `focalLength=50`, `horizontalAperture=20.955`, `verticalAperture=15.2908`; at `960x720`, approximately `fx=2290.623`, `fy=2354.357`, `fov_x=23.670 deg`, `fov_y=17.387 deg` | OpenUSD fallback `clippingRange=(1, 1000000)` scene units | Whatever transform is authored on the camera prim; no robot mount default | USD cameras look down local `-Z`, with `+Y` up and `+X` right |
| `xiaotao_legacy_override` | Historical local test override, removed from active source during branch cleanup | `640x480` | `fov_y=70 deg`, `fov_x=86.067 deg`, `fx=fy=342.756`, `cx=320`, `cy=240` | Renderer defaults unless overridden | `camera_z = occupancy_upper_z - 0.098 m` | Same path look-ahead/look-down policy as `navdp_legacy_fpv` |
| `gradio_demo_intrinsics` | Historical demo/debug override, removed from active source during branch cleanup | `640x480` | `fx=fy=360`, `cx=350`, `cy=230`, approximately `fov_x=83.267 deg`, `fov_y=67.380 deg` | Demo-specific | Not a robot rig | Not a production default |

## G1 Notes

The local renderer already treats G1 as the default peer-robot visual asset:

- GLB: `assets/robots/g1_29dof_mode_16.glb`
- robot bounds fallback: radius `0.3 m`, height `1.3 m`
- robot model convention in the G1 follow example: GLB up axis `+Z`, robot
  forward axis `+X`

Unitree's public G1 product page lists the standing dimensions as
`1320x450x200 mm` and the sensing sensor class as `Depth Camera+3D LiDAR`, but
does not publish camera intrinsics or mount transforms. Therefore
`g1_head_fpv_default` is intentionally provisional. Replace it with the imported
G1 USD/OpenUSD sensor rig as soon as that file is available.

## Local Source Comparison

- `render_label_paths.py` and `render_label_paths_telesim.py` both default to
  `height_offset=0.3`, `resolution=960 720`, `fov_deg=70`, `znear=0.001`,
  `zfar=30`, `look_ahead=2`, and `look_down=0.1`.
- `utils/telesim_path_json_outputs.py` serializes the same pinhole intrinsics
  from vertical FOV, aspect ratio, and centered principal point.
- `run_random_fpv_datagen*.sh` and `run_random_human_datagen*.sh` preserve
  `HEIGHT_OFFSET=0.3`.
- The removed Xiaotao local test fork used `HEIGHT_OFFSET=-0.098`,
  `RES_W=640`, `RES_H=480`, `FOV=70`, `LOOK_AHEAD=2`, and `LOOK_DOWN=0.1`.
- `scripts/run_g1_robot_follow_example.py` renders the base GS frames at
  `960x720`, uses a `1.5 m` follow distance, and overlays the G1 GLB using
  per-frame camera metadata instead of defining a physical camera rig.

## Implementation TODOs

- [x] Add the profile names above to the MassGen sensor-rig schema.
- [x] Make `navdp_legacy_fpv` the default for generated
  manifests that do not request a robot-mounted sensor.
- [x] Make `g1_head_fpv_default` selectable by name for G1 jobs and clearly mark
  it as provisional in run summaries.
- [ ] Add a preflight warning when raw OpenUSD fallback optics are used because
  authored camera optics are missing.
- [x] Add regression tests that compute the derived intrinsics for
  `navdp_legacy_fpv`, `g1_head_fpv_default`, and `openusd_camera_fallback`.

## External References

- OpenUSD `UsdGeomCamera` documents the camera coordinate convention and fallback
  camera attributes: `focalLength=50`, `horizontalAperture=20.955`,
  `verticalAperture=15.2908`, and `clippingRange=(1, 1000000)`.
  <https://openusd.org/release/api/class_usd_geom_camera.html>
- Isaac Sim camera sensors wrap or create camera prims and expose resolution,
  world/local transform, annotators, projection type, and batched camera views.
  <https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.sensors.camera/docs/index.html>
- Isaac Sim sensor docs group camera/depth sensors, RTX sensors, and physics
  sensors as separate sensor categories for robot setups.
  <https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_camera.html>
- Unitree's public G1 page lists standing dimensions and sensing sensor class,
  but not camera intrinsics or mount transforms.
  <https://www.unitree.com/g1/>
