# MassGen Render Manifest Contract

Date: 2026-08-04
Branch: `massgen`

This contract is the boundary between `Navdp_Datagen_Pathplanner` scenario JSONs
and `Navdp_Datagen` rendering workers. Pathplanner owns mission construction.
The renderer owns Gaussian/GLB composition, camera metadata, visibility culling,
video output, and GPU scheduling.

## CLI

```bash
python3 scripts/massgen/export_massgen_render_manifest.py \
  --scenario-json /path/to/pathplanner_scenario.json \
  --action-catalog-json /path/to/action_codex.json \
  --output-json /path/to/render_manifest.json
```

For self-service runs with sensor selection and preflight, use:

```bash
scripts/massgen/prepare_render_run.py \
  --config-json configs/massgen/render_run_example.json \
  --summary
```

For frame-interest selection before rendering, write a selection manifest and
optionally a selected-window render manifest:

```bash
python3 scripts/massgen/select_frame_interest_windows.py \
  --manifest-json /path/to/render_manifest.json \
  --target-count 1000 \
  --output-selection-json /path/to/selection_manifest.json \
  --output-render-manifest-json /path/to/selected_render_manifest.json
```

`render_manifest_jobs.py` can also apply the selection manifest directly:

```bash
python3 scripts/massgen/render_manifest_jobs.py \
  --manifest-json /path/to/render_manifest.json \
  --frame-selection-json /path/to/selection_manifest.json \
  --write-inputs
```

Defaults:

- render backend: `gsplat`
- FPS: `10.0`
- human and peer-robot visibility culling: enabled in generated jobs
- default peer robot GLB: `assets/robots/g1_29dof_mode_16.glb`

## Avatar Action Sources

The manifest supports two action-source modes.

`configs/massgen/avatar_action_generation_inputs.json` is the batch input
matrix for materializing missing avatar clips before a render run. It contains:

- reusable action templates with Kimodo/STMC prompts, optional keypoints, root
  motion policy, duration, and quality checks;
- avatar identities with appearance prompts and canonical asset placeholders;
- per-avatar required and nice-to-have action lists by mission family;
- expanded `generation_requests`, one per avatar/action pair, with concrete
  output paths under `assets/human_actions/generated/{human_resource_id}/{action_id}`.

Pre-generated action assets:

- `asset.pre_generated: true`
- `asset.requires_generation: false`
- `asset.ply_frame_dir` or `asset.manifest_path` points at renderer-ready
  action frames

Generated action requests:

- `asset.pre_generated: false` or `source` is `kimodo`, `stmc`, or
  `generated_on_the_fly`
- `asset.requires_generation: true`
- `generation_request.enabled: true`
- `generation_request.generator` records `kimodo`, `stmc`, or the configured
  generator source
- `generation_request.instruction` preserves the text prompt
- `generation_request.input_style` is `text` or `text_with_keypoints`
- `generation_request.keypoints` preserves optional keypoint or waypoint
  constraints
- `generation_request.output_contract` names where the generated manifest,
  PLY frames, and SMPL-X frames should be materialized before rendering

Server-side preflight should fail a render job if any action segment still has
`asset.requires_generation: true`. The action generation platform should fill
the output contract first, then rerun manifest conversion or patch the action
segment into a pre-generated asset.

## Top-Level Shape

The generated JSON has:

- `source`: original scenario id, scene id, schema version, and source path;
- `render_backend`: currently defaults to `gsplat`;
- `mission_families`: mission types present in the scenario;
- `social_law_ids`: union of mission and social-structure law ids;
- `scene_assets`: copied from the Pathplanner scenario;
- `timing`: FPS, start/end time, and frame count;
- `render_layers`: scene Gaussian, human Gaussian PLY sequence, and peer-robot
  GLB overlay settings;
- `actors.humans`: normalized human poses, trajectories, action segments,
  mission bindings, appearance records, and visibility bounds;
- `actors.robots`: normalized robot pose tracks, GLB asset records, embodiment
  records, and visibility bounds;
- `missions`: renderer-relevant mission fields plus original mission metadata;
- `events`: normalized scenario event log;
- `navigation_supervision`: deterministic scenario-level instruction and
  decision-label provenance copied from Pathplanner;
- `rendering_metadata_contract`: declares complete-path availability,
  stationary-sample preservation, metadata paths, and consumer-defined frame
  selection;
- `jobs`: one render job per active/training robot viewpoint;
- `warnings`: non-fatal conversion warnings for missing assets or unsupported
  families.

## Frame-Interest Selection

The frame-interest selector is a Datagen-side step between Pathplanner manifest
export and rendering. It chooses target frames for training/testing, expands
each selected target to the required `32 past + current` window, and passes only
those window jobs to the renderer. Future frames are not packaged.

The canonical v0.5 selector is event-aware: it keeps sparse representatives of
important contiguous semantic episodes and all mandatory anchors. Fixed target
budgets remain available only for controlled comparisons; they are not a
per-path cap and do not replace the canonical semantic selection.

Selection manifests use schema `navdp_frame_interest_selection/v0.1` and store:

- source manifest records and fingerprints;
- adaptive interest-bucket and secondary action distributions;
- selected chunks with target frame, renderer-local target index, action,
  bucket, score, reasons, navigation signals, and source frame indices;
- a render contract requiring all 33 window frames to be rendered;
- a render frame index with unique source frames for future sparse rendering;
- frame-level navigation-signal counts and contiguous-event coverage summaries.

Large policy surveys may store the same renderer-independent chunk fields in
gzip-compressed JSON Lines (`navdp_poi_survey_selection/v1.0`), one cohort per
line. Each record keeps source manifest/job identity, the complete-path sample,
target frame and target-window index, 33 source indices, semantic signals,
target role, and mandatory-anchor types; downstream consumers can construct
their own render jobs without replaying POI scoring.

The selected-window render manifest keeps the original source manifest metadata
but replaces `jobs` with one job per selected frame of interest. Each selected
job:

- has `job_id` suffixed with `__m<source_manifest_index>__foi_<target_frame>`;
- carries only the selected 33 camera trajectory samples;
- rewrites renderer-local frame/sample ids to `0..32`;
- declares `target_window_index=32`, identifying the current frame as the final
  sample in the default past-only window;
- preserves source frame indices in point and camera metadata;
- sets `camera.preserve_frame_samples=true`.

Source jobs carry `frame_catalog.trajectory_scope=complete_source_path` and can
be rendered in full or filtered by any downstream policy. Selected jobs carry
`frame_catalog.trajectory_scope=selected_past_context_window`; their complete
robot tracks remain available under `actors.robots[*].trajectory`. The built-in
selector is therefore one supported consumer, not a restriction in the dataset
format.

Render executors and label-path helpers must honor
`camera.preserve_frame_samples` and `metadata.preserve_frame_samples`. This
prevents stop/yield/repeated-pose context from being removed by trajectory
deduplication or distance resampling. Executor planning also blocks
`--minimal-frames` values that would truncate selected frame-interest windows.
All MassGen camera jobs set this flag, including full-path jobs. If exported
trajectory samples have gaps in their frame indices, the executor interpolates
one camera timestamp per missing frame before building camera, human, and peer
robot tracks. Repeated camera positions remain repeated render frames so scene
actors can continue moving while the ego robot is stopped.

Self-service run extension:

- `sensor_rigs`: normalized robot-mounted sensor definitions imported from an
  Isaac Sim/OpenUSD-compatible robot rig or an exported rig JSON, including
  sensor names, robot-relative transforms, camera intrinsics, resolution,
  clipping range, rate, and requested output modalities;
- `jobs[*].sensors`: named sensor selections for each robot viewpoint, replacing
  hard-coded FPV/follow camera constants while keeping a default FPV camera for
  backwards compatibility;
- fallback sensor profile names documented in
  `docs/camera_sensor_defaults.md`, including `navdp_legacy_fpv`,
  `g1_head_fpv_default`, and `openusd_camera_fallback`.

## Mission Family Mapping

| Family | Viewpoint Jobs | Human Action Handling | Peer Robots |
| --- | --- | --- | --- |
| `deliver_to_human` | Assigned robot | Target human gets `receive_item` mission hint; bystanders remain visible/cullable | None unless scenario has extra robots |
| `serve_queue` | Assigned robot | Queue targets and queue structures get `queue_wait` hints | None unless scenario has extra robots |
| `mission_stream` | Explicit `training_robot_ids`/`active_robot_ids`, else all robots | Child mission targets and social structures preserved | All non-ego robots listed per job |
| `human_guided_uncertain_region` | Assigned robot | Informant/guidance actors infer `wave` | None unless scenario has extra robots |
| `dense_dynamic_avoidance` | Assigned robot, or all robots if marked active | Moving humans infer `walk`; waiting actors infer stationary actions | Other active robots listed per job |
| `dense_dynamic_humans` | Assigned robot | Moving humans infer `walk`; all humans remain cullable | None unless scenario has extra robots |
| `dense_multi_robot` | Explicit active/training ids, else all robots | Usually no humans | All non-ego robots listed per job |
| `dense_dynamic_combined` | Explicit active/training ids, else all robots | Moving humans infer `walk`; queues/groups retain action hints | All non-ego robots listed per job |
| `navigate_with_social_constraints` | Assigned robot | Queue, pedestrian-flow, and group structures map to queue/walk/gesture hints | None unless scenario has extra robots |

Schema-only Pathplanner families are converted with warnings only:

- `human_guided_person_disambiguation`
- `human_guided_route_correction`

## Current Limits

- The manifest is declarative; it does not launch CUDA rendering.
- Generated actions are represented as requests. They must be materialized into
  renderer-ready PLY/SMPL-X frame directories before GPU rendering.
- Missing human PLY frame directories are warnings today when no generation
  request is attached. The server launcher should turn those into preflight
  failures before reserving a GPU.
- Peer robots are referenced in jobs, but multi-robot GLB compositing still
  needs a manifest-driven executor.
- Human action switching is represented as multiple `action_segments`; the hot
  render path still needs multi-human, multi-action composition.
- Sensor settings are still script defaults today. The manifest needs a
  normalized sensor-rig import path before users can render with arbitrary
  Isaac Sim/OpenUSD robot sensor setups. Until then, default and comparison
  profiles are documented in `docs/camera_sensor_defaults.md`.
- Frame-interest selection currently materializes one 33-frame render job per
  selected target. The manifest already reports unique source frame reuse, but a
  renderer that renders each unique source frame only once is still future work.
