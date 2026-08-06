# MassGen Self-Service Render Run

Date: 2026-08-05
Branch: `massgen`

This is the user-facing entry point for preparing a MassGen render run without
editing renderer scripts.

## Prepare and Preflight

```bash
scripts/massgen/prepare_render_run.py \
  --config-json configs/massgen/render_run_example.json \
  --preflight-only \
  --summary
```

The command:

- converts a Pathplanner scenario JSON into a render manifest;
- attaches a selected sensor rig or fallback camera profile;
- validates input files, output paths, robot assets, and selected sensors;
- prints a compact summary with job count, robot viewpoints, sensors, frame
  count, warnings, and blocking errors.

Add `--write` to write the generated manifest and optional summary JSON:

```bash
scripts/massgen/prepare_render_run.py \
  --config-json configs/massgen/render_run_example.json \
  --write
```

The example config writes to `out/massgen_runs/example/`, which is ignored by
Git.

## Config Fields

Required:

- `scenario_json`: Pathplanner scenario JSON.
- `output_root`: render output root.

Optional:

- `action_catalog_json`: Pathplanner action catalog with an `actions` list.
- `manifest_json`: output manifest path; defaults to
  `<output_root>/render_manifest.json`.
- `summary_json`: optional run-summary path written with `--write`.
- `sensor_profile`: fallback profile name. Supported values:
  - `navdp_legacy_fpv`
  - `g1_head_fpv_default`
  - `openusd_camera_fallback`
- `sensor_rig_json`: imported normalized or Isaac Sim-style sensor rig JSON.
  This takes precedence over `sensor_profile`.
- `selected_sensors`: sensor names to attach to each render job.
- `strict_assets`: when `true`, missing scene/robot/action assets block the run;
  when `false`, they are warnings.
- `gpu_devices`, `workers`, `fps`, `render_backend`,
  `default_robot_glb`, and visibility-culling margins.

## Imported Sensor Rig JSON

Minimal imported rig:

```json
{
  "rig_id": "g1_usd_rig",
  "robot_id": "robot_alpha",
  "format": "isaacsim_export_json",
  "cameras": [
    {
      "camera_name": "front_rgb",
      "camera_prim_path": "/World/G1/head/front_rgb",
      "local_position": [0.1, 0.0, 1.2],
      "local_rotation_rpy_deg": [0.0, -5.0, 0.0],
      "resolution": [960, 720],
      "fov_y_deg": 70.0,
      "clipping_range": [0.05, 30.0],
      "modalities": ["rgb", "depth"]
    }
  ]
}
```

`sensor_rig_json` can use `sensors` or `cameras`. The importer normalizes camera
name, prim path, robot-relative transform, intrinsics, clipping range, rate, and
modalities into `manifest.sensor_rigs`.

## Current Boundary

This command prepares and validates render manifests. It does not launch GPU
render workers yet. The next step is to connect `summary.status == "ready"` to
the server scheduler/renderer launcher.
