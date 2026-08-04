# MassGen Render Manifest Contract

Date: 2026-08-04
Branch: `massgen`

This contract is the boundary between `Navdp_Datagen_Pathplanner` scenario JSONs
and `Navdp_Datagen` rendering workers. Pathplanner owns mission construction.
The renderer owns Gaussian/GLB composition, camera metadata, visibility culling,
video output, and GPU scheduling.

## CLI

```bash
python3 scripts/export_massgen_render_manifest.py \
  --scenario-json /path/to/pathplanner_scenario.json \
  --action-catalog-json /path/to/action_codex.json \
  --output-json /path/to/render_manifest.json
```

Defaults:

- render backend: `gsplat`
- FPS: `10.0`
- human and peer-robot visibility culling: enabled in generated jobs
- default peer robot GLB: `assets/robots/g1_29dof_mode_16.glb`

## Avatar Action Sources

The manifest supports two action-source modes.

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
- `jobs`: one render job per active/training robot viewpoint;
- `warnings`: non-fatal conversion warnings for missing assets or unsupported
  families.

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
