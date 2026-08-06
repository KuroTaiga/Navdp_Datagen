# Per-Mission-Family Rendering Handoff

Date: 2026-08-06
Branch: `massgen`
Starting point before this handoff doc: `9f1cfa6`

This is the fresh-context handoff for implementing MassGen rendering one mission
family at a time. Read this before editing code.

## Current State

`Navdp_Datagen` is now clean enough to move on from cleanup:

- working tree was clean before creating this handoff;
- no untracked or ignored first-party script-like files remain;
- root launcher duplication was reduced by moving shared storage/runtime logic
  into `scripts/render/launchers/legacy_storage_runner.sh`;
- shell helpers were made compatible with macOS Bash 3.2;
- private `scripts/.env` was removed from Git tracking and replaced by
  `scripts/slave_finder.example.env`.

Do not spend the next chat on more cleanup unless it directly blocks
mission-family rendering.

## Goal

Build the render side of MassGen so a user can render wanted data by mission
family without editing renderer internals:

1. choose or generate scenarios by mission family;
2. attach a robot sensor profile or imported Isaac Sim/OpenUSD-style rig;
3. preflight assets and output paths;
4. render one job per viewpoint robot;
5. handle humans, action segments, visibility culling, and peer robots according
   to the family.

## Repos

Renderer/datagen repo:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen
```

Pathplanner/MassGen source repo:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner
```

Pathplanner owns CPU-side scenario construction. This repo owns render
preflight, manifests, Gaussian/GLB composition, cameras, videos, and scheduling.

## Mission Families

Active Pathplanner MassGen families:

- `deliver_to_human`
- `serve_queue`
- `mission_stream`
- `human_guided_uncertain_region`
- `dense_dynamic_avoidance`
- `dense_dynamic_humans`
- `dense_multi_robot`
- `dense_dynamic_combined`
- `navigate_with_social_constraints`

Active social-navigation subfamilies:

- `navigate_with_social_constraints:personal_space`
- `navigate_with_social_constraints:pedestrian_yield`
- `navigate_with_social_constraints:group_integrity`
- `navigate_with_social_constraints:queue_order`

Schema-only, not active builders yet:

- `human_guided_person_disambiguation`
- `human_guided_route_correction`

Do not add renderer-specific assumptions for the schema-only families until
Pathplanner has active builders and examples for them.

## Rollout Order

Start easy and validate each family before moving on.

### Easy: One Robot, Stationary or Near-Stationary Humans

1. `deliver_to_human`
2. `serve_queue`
3. `human_guided_uncertain_region`
4. `navigate_with_social_constraints:personal_space`
5. `navigate_with_social_constraints:queue_order`
6. `navigate_with_social_constraints:group_integrity`

Expected render features:

- one viewpoint robot;
- default FPV/G1 sensor profile works;
- target/queue/informant/group humans are mostly stationary;
- action hints should map to `receive_item`, `queue_wait`, `wave`, or `stand`;
- no peer-robot rendering required unless the scenario unexpectedly includes
  extra robots.

### Medium: One Robot, Moving Humans

7. `navigate_with_social_constraints:pedestrian_yield`
8. `dense_dynamic_humans`
9. `dense_dynamic_avoidance` when configured as one robot plus moving humans

Expected render features:

- one viewpoint robot;
- moving human trajectories and `walk` action segments;
- per-frame visibility culling before merging human PLYs;
- debug counters for candidate/visible/culled humans should be added before
  trusting dense cases.

### Hard: Multiple Robots and Multiple Viewpoints

10. `dense_multi_robot`
11. `dense_dynamic_combined`
12. `mission_stream`
13. `dense_dynamic_avoidance` when active peer robots are present

Expected render features:

- one render job per active/training robot;
- each job has its own selected sensors;
- peer robots must be visible from each viewpoint when in frame;
- peer robot GLB overlay needs manifest-driven multi-robot pose support;
- depth composition and culling are required before using hard-family outputs
  for training.

## Current Implemented Boundary

Manifest conversion and self-service preflight already exist.

Key files:

- `utils/massgen_render_manifest.py`
- `scripts/massgen/export_massgen_render_manifest.py`
- `navdp_datagen/massgen/run_config.py`
- `scripts/massgen/prepare_render_run.py`
- `navdp_datagen/sensors.py`
- `configs/massgen/render_run_example.json`
- `configs/massgen/example_scenario.json`
- `configs/massgen/example_action_catalog.json`
- `docs/massgen_render_manifest_contract.md`
- `docs/massgen_self_service_render_run.md`
- `docs/camera_sensor_defaults.md`

Current behavior:

- scenario JSONs convert to render manifests;
- `render_backend` defaults to `gsplat`;
- jobs are generated per active/training robot;
- peer robot pose tracks are represented in job metadata;
- human action segments and generation requests are represented;
- sensor rigs are attached from fallback profiles or imported rig JSON;
- preflight validates paths, writable outputs, selected sensors, and strict
  asset behavior.

Important limit:

`prepare_render_run.py` prepares manifests and summaries. It does not launch
GPU render workers yet. The next code should build the manifest-driven render
executor.

## Current Tests

Useful local checks:

```bash
python3 -m pytest \
  tests/test_massgen_render_manifest.py \
  tests/test_massgen_render_run_config.py \
  tests/test_sensor_profiles.py \
  tests/test_actor_visibility.py
```

Self-service preflight smoke:

```bash
scripts/massgen/prepare_render_run.py \
  --config-json configs/massgen/render_run_example.json \
  --preflight-only \
  --summary
```

Expected status for the example config is `ready` with warnings when demo scene
assets are missing and `strict_assets` is `false`.

## Scenario Inputs

The intended local destination for generated per-family scenario examples is:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen/tests/inputs
```

At this handoff, `tests/inputs` exists but does not contain the requested five
scenarios per mission family. The fresh-context chat should either generate or
copy those inputs before implementing family-specific rendering tests.

Pathplanner generation entry points:

- `Code/navdp/benchmark/cli.py`
- `Code/navdp/benchmark/mass_generation.py`
- `scripts/run_mass_generation_platform.sh`
- `scripts/run_mass_generation_container.sh`
- `docs/human_centric_mass_generation.md`

Pathplanner command shape for small generation runs:

```bash
cd /Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner
python3 -m Code.navdp.benchmark.cli generate-mass-examples \
  --output-dir /Users/dongjk/ProjectFiles/Navdp_Datagen/tests/inputs/massgen_generated \
  --manifest /Users/dongjk/ProjectFiles/Navdp_Datagen/tests/inputs/massgen_generated/mass_example_manifest.json \
  --scene-exclusions configs/data_quality/scene_exclusions.yaml \
  --workers 2 \
  --target-valid-per-type 5 \
  --max-candidates-per-type 100 \
  --progress-every 1 \
  --human-agent-mode rule \
  --mission-type deliver_to_human
```

For social-navigation subfamilies, add:

```bash
--mission-type navigate_with_social_constraints \
--social-nav-law-case personal_space
```

Run one family/subfamily at a time so failures and renderer coverage stay
isolated. Use the Pathplanner exclusion registry so banned scenes are not
selected:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner/configs/data_quality/scene_exclusions.yaml
```

After generation, validate scenario files with Pathplanner before feeding them
to this repo:

```bash
python3 -m Code.navdp.benchmark.cli validate path/to/scenario.json
```

## Renderer Implementation Plan

### 1. Add a Manifest-Driven Executor

Recommended new module:

```text
navdp_datagen/massgen/render_executor.py
```

Recommended CLI:

```text
scripts/massgen/render_manifest_jobs.py
```

Initial responsibilities:

- load a render manifest;
- select jobs by mission family, job id, robot id, or sensor name;
- validate `summary.status == ready` or rerun preflight;
- map each job into existing renderer arguments;
- force `GAUSSIAN_RENDER_BACKEND=gsplat`;
- write per-job output directories and metadata;
- support `--dry-run`, `--limit`, `--job-id`, and `--family`.

Do not start with all families. Make the executor work for one easy family
first.

### 2. Start With `deliver_to_human`

Acceptance criteria:

- five generated `deliver_to_human` scenarios exist under `tests/inputs`;
- `prepare_render_run.py` produces one job per scenario;
- strict preflight fails clearly if target human action assets are missing;
- non-strict preflight lists missing action/scene assets as warnings;
- executor dry-run prints the exact render command/job plan;
- server render smoke shows target human visible from robot FPV.

Likely render data needed from manifest:

- `jobs[*].viewpoint_robot_id`
- `jobs[*].trajectory`
- `jobs[*].sensors`
- `actors.humans[*].action_segments`
- `actors.humans[*].mission_bindings`
- `scene_assets.splat_model_path`
- `timing.fps`, `timing.frame_count`

### 3. Then `serve_queue`

Acceptance criteria:

- queue participants are included in `actors.humans`;
- queue action hints map to `queue_wait`;
- render metadata records queue/social structure ids;
- visual smoke confirms queue actors are placed and visible/cullable.

### 4. Then `human_guided_uncertain_region`

Acceptance criteria:

- informant/guidance actor uses `wave` or a configured guidance action;
- event-aligned metadata is preserved in output;
- render smoke confirms informant visibility near the guidance event.

### 5. Medium Families

Before moving-human families, add:

- multi-action segment materialization for humans;
- per-frame human visibility culling before PLY merge;
- debug counters per frame/job:
  - candidate humans;
  - visible humans;
  - culled humans;
  - merged PLY point counts.

### 6. Hard Families

Before multi-robot families, add:

- manifest-driven GLB overlay for multiple peer robots;
- one pose track per peer robot per job;
- peer robot visibility culling before pyrender work;
- optional depth composition against saved GS depth maps;
- output metadata listing rendered and culled peer robots per frame.

Relevant existing files:

- `utils/glb_robot_compositor.py`
- `scripts/render/assets/render_glb_robot_overlay.py`
- `tests/test_glb_robot_compositor.py`

## Sensor Defaults

Use existing profiles unless a scenario/run config imports a rig:

- `navdp_legacy_fpv`
- `g1_head_fpv_default`
- `openusd_camera_fallback`

Reference:

```text
docs/camera_sensor_defaults.md
```

Do not reintroduce hard-coded camera constants in family-specific shell scripts.
The family executor should read sensors from manifest jobs.

## Known Caveats

- Local macOS cannot validate CUDA rendering. GPU validation must run on the
  render server.
- The current renderer hot path still needs manifest-driven multi-human,
  multi-action composition.
- Generated Kimodo/STMC action requests are represented in manifests but must be
  materialized into renderer-ready assets before strict rendering.
- Peer robots are represented in manifest jobs, but multi-peer GLB composition
  is not fully connected to the render executor yet.
- `release/navdp_path_renderer/` is a release mirror. Do not update it during
  the first per-family implementation unless the user explicitly asks for a
  release package refresh.

## Suggested First Prompt For Fresh Chat

```text
Read docs/per_mission_family_rendering_handoff.md. Start per-mission-family
rendering with deliver_to_human. First verify or generate five Pathplanner
scenario JSONs under tests/inputs, then implement the smallest manifest-driven
executor dry-run that maps those jobs to existing renderer inputs without
changing family behavior.
```
