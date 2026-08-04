# MassGen Generation Pipeline TODOs

Date: 2026-08-04
Branch: `massgen`

This document tracks the implementation work for turning Pathplanner MassGen
scenario families into rendered NavDP training data.

## Current Direction

Use `Navdp_Datagen_Pathplanner` as the mission/scenario source while
`Navdp_Datagen` owns rendering, actor composition, camera export, video frames,
and GPU scheduling.

The renderer must support:

- dynamic humans with action clips beyond standing/walking;
- multiple robots per scenario;
- one rendered viewpoint stream per active robot;
- other robots visible from each robot camera;
- actor visibility culling so off-camera human PLYs are not merged into the
  Gaussian scene;
- multi-GPU server scheduling.

## Mission Family Rendering Needs

| Family | Scenario Needs | Rendering Needs |
| --- | --- | --- |
| `deliver_to_human` | Target human identity and interaction endpoint | Dynamic target human, optional nearby humans, robot FPV stream |
| `serve_queue` | Queue actors and service point | Queue actions, waiting/service actions, robot approach stream |
| `mission_stream` | Multiple robots, mission dispatch, child missions | One stream per robot, render peer robots when visible, route/event metadata |
| `human_guided_uncertain_region` | Informant human and guidance event trace | Human talk/gesture action, robot FPV stream, event-aligned metadata |
| `dense_dynamic_avoidance` | Dense moving humans and/or robots | Dynamic actor crowd, visibility culling, collision/debug traces |
| `dense_dynamic_humans` | Many moving humans | Dynamic action clips, per-frame visible-actor selection |
| `dense_multi_robot` | Multiple robot trajectories | Per-robot viewpoint streams, peer-robot GLB/mesh rendering |
| `dense_dynamic_combined` | Dense humans plus multiple robots | Combined dynamic humans, robot overlays, visibility culling |
| `navigate_with_social_constraints` | L1-L4 social-law subfamilies | Law-case-specific humans/groups/queues/flows and violation/reference metadata |

Schema-only Pathplanner mission types not active in MassGen yet:

- `human_guided_person_disambiguation`
- `human_guided_route_correction`

Do not add renderer assumptions for these until Pathplanner adds active
builders and scenario examples.

## Workstreams

### 1. Scenario Render Contract

- [x] Define a portable scenario-to-render manifest generated from Pathplanner
  scenario JSONs.
- [x] Normalize actor records into:
  - actor id;
  - actor kind: `human` or `robot`;
  - asset reference;
  - action/animation reference;
  - per-frame or time-sampled world pose;
  - visibility radius/height;
  - mission/event bindings.
- [x] Normalize robot viewpoint records into one render job per robot.
- [x] Preserve source scenario id, mission id, robot id, and frame timestamps in
  camera/action metadata.

Implemented in:

- `utils/massgen_render_manifest.py`
- `scripts/export_massgen_render_manifest.py`
- `docs/massgen_render_manifest_contract.md`

### 2. Dynamic Human Actions

- [x] Map Pathplanner human `behavior_timeline` actions to available Gaussian
  PLY action sequences.
- [x] Add an action catalog lookup layer for sit, stand, walk, wait, queue,
  gesture, guidance, and service interactions.
- [x] Allow one human to switch action sequences over time.
- [x] Keep action-time sampling deterministic from scenario seed and actor id.
- [x] Preserve pre-generated action assets and Kimodo/STMC-style text or
  text-plus-keypoint generation requests in the render manifest.
- [ ] Validate missing action assets fail clearly before GPU rendering starts.

### 3. Multi-Robot Rendering

- [x] Generate one FPV/follow render job per active robot trajectory.
- [x] Export peer-robot pose tracks for every viewpoint stream.
- [ ] Reuse the existing GLB robot compositor for peer robots.
- [ ] Add multi-robot overlay support instead of one `--poses-json` per run.
- [ ] Depth-compose peer robots against saved GS depth maps when available.
- [ ] Record which peer robots were rendered or culled per frame.

### 4. Visibility Culling

Started in this branch:

- [x] Added `utils/actor_visibility.py` for CPU-side bounding-sphere frustum
  checks.
- [x] Added `--actor-visibility-culling` to `render_label_paths_telesim.py` for
  the existing single Gaussian actor path.
- [x] Added local non-CUDA tests for the visibility math.
- [x] Render manifests now enable human and peer-robot visibility culling by
  default for generated jobs.

Remaining:

- [ ] Apply the same culling to scenario bystander humans before
  `apply_transform_to_frame`.
- [ ] Apply culling to GLB robot overlays before pyrender work.
- [ ] Add debug counters per frame/job:
  - candidate actors;
  - visible actors;
  - culled actors;
  - actor PLY points merged.
- [ ] Add optional conservative margin per actor kind.

### 5. Multi-GPU Server Scheduling

- [ ] Add a server launcher that shards render jobs across CUDA devices.
- [ ] Keep one process bound to one GPU by setting `CUDA_VISIBLE_DEVICES`.
- [ ] Track per-GPU active jobs, failures, CUDA OOM, and retry counts.
- [ ] Keep MassGen CPU scenario construction separate from GPU rendering.
- [ ] Make `GAUSSIAN_RENDER_BACKEND=gsplat` the default in server render
  launchers.
- [ ] Add a platform smoke command for one scene, one mission family, and all
  robots.

### 6. Validation

Local macOS:

- [x] `python3 -m py_compile __init__.py render_label_paths_telesim.py utils/actor_visibility.py tests/test_actor_visibility.py`
- [x] `/Users/dongjk/miniconda3/bin/python3.13 -m pytest tests/test_actor_visibility.py`
- [x] `python3 -m py_compile utils/massgen_render_manifest.py scripts/export_massgen_render_manifest.py tests/test_massgen_render_manifest.py`
- [x] `/Users/dongjk/miniconda3/bin/python3.13 -m pytest tests/test_massgen_render_manifest.py tests/test_actor_visibility.py`
- [x] Scenario-manifest conversion tests with tiny fixture JSONs.
- [x] CLI smoke conversion against Pathplanner
  `minimal_passing_mission_stream.json`.

Server/platform:

- [ ] One `deliver_to_human` render job with visible target human.
- [ ] One `serve_queue` render job with queue actors.
- [ ] One `dense_multi_robot` render with at least two robot viewpoints.
- [ ] One `dense_dynamic_combined` render with humans plus peer robots.
- [ ] Confirm logs show `gsplat` backend selection.
- [ ] Confirm no off-camera human PLYs are merged when culling is enabled.

## Next Implementation Step

Build the manifest-driven render executor. It should materialize each job into
camera frames, compose visible human Gaussian PLY action segments, render/cull
peer GLB robots, and launch with `GAUSSIAN_RENDER_BACKEND=gsplat` on the server.
