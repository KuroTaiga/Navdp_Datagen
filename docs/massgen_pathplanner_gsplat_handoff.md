# MassGen Pathplanner and gsplat Handoff

Date: 2026-07-31

This is a planning and resume document only. No runtime code has been changed in
this pass.

## Goal

1. Ensure Gaussian-splat rendering uses `gsplat` as the rendering backend.
2. Respect the local-machine constraint: this machine is macOS and cannot run
   CUDA render tests. CUDA/GPU validation must run on the calculation/rendering
   platform.
3. Update or integrate MassGen so it can handle the mission families present in:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner
```

## Repo State Observed

Current renderer/datagen repo:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen
HEAD: 20d78c7
working tree: clean before this documentation file
```

Pathplanner repo inspected:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner
HEAD: 288a428
working tree: dirty
```

Pathplanner dirty files at inspection time:

- `.DS_Store`
- `Code/connect_A_B.py`
- `Code/navdp/benchmark/example_generation.py`
- untracked generated outputs under `tmp/`
- untracked generated reports under `reports/mass_generation/`

The dirty source edits are performance-related and should be reviewed before
porting or copying:

- `Code/connect_A_B.py`: A* uses fixed neighbor step costs and a closed set.
- `Code/navdp/benchmark/example_generation.py`: optimizes nearest clear-cell
  lookup, reachable-cell reuse, mission-stream trajectory reuse, social-law
  path reuse, and some L1/L3 route-pool logic.

## Key Finding

`Navdp_Datagen` is currently the rendering/datagen repo. It does not contain
`Code/navdp/benchmark/mass_generation.py`, `Code/navdp/benchmark/cli.py`, or
`tests/test_mass_generation.py`.

`Navdp_Datagen_Pathplanner` contains the MassGen implementation, benchmark
schema, mission-family logic, platform scripts, Dockerfile, and tests.

Therefore the MassGen work is not a narrow edit to an existing local module in
`Navdp_Datagen`. It is either:

1. keep MassGen in Pathplanner as the source of truth and make this repo consume
   Pathplanner scenario/path artifacts for rendering, or
2. port the Pathplanner MassGen package and its dependencies into this repo.

Do not start by hunting for a local MassGen module in `Navdp_Datagen`; it is not
there.

## Rendering Backend Findings

Relevant files in `Navdp_Datagen`:

- `gaussian_renderer/__init__.py`
- `render_label_paths.py`
- `render_label_paths_telesim.py`
- `release/navdp_path_renderer/render_path.py`
- `release/navdp_path_renderer/README.md`
- `release/navdp_path_renderer/requirements.txt`
- `release/navdp_path_renderer/TeleSim3D/tele_sim/rendering/gaussian_backend.py`

Current backend behavior:

- `gaussian_renderer._prefer_gsplat()` defaults to `gsplat` via
  `GAUSSIAN_RENDER_BACKEND=gsplat`.
- `gaussian_renderer.render()` uses `_gsplat_render()` when:
  - `separate_sh` is false,
  - `pipe.compute_cov3D_python` is false,
  - `scaling_modifier == 1.0`.
- `render_label_paths_telesim.py` calls `gaussian_renderer.render`, so the
  normal TeleSim FPV/follow render path should use `gsplat` by default.
- `release/navdp_path_renderer/render_path.py` exposes `--backend gsplat` and
  writes `GAUSSIAN_RENDER_BACKEND` into the subprocess environment.
- `release/navdp_path_renderer/README.md` already documents that render-only
  usage defaults to `gsplat`, and `release/navdp_path_renderer/requirements.txt`
  includes `gsplat`.

Important gap:

- `gaussian_renderer.render_or(..., orthographic=True)` still skips `gsplat` and
  falls back to the legacy `diff_gaussian_rasterization` path.
- Root `environment.yml` did not show `gsplat` in the targeted dependency
  search, even though the release package requires it.

Decision needed:

- If "rendering" means FPV/follow production rendering, the existing default is
  already `gsplat` for the main perspective path.
- If "rendering" means every Gaussian render call, including orthographic BEV
  or room/topdown utilities, then `render_or` needs a `gsplat` orthographic path
  or those utilities must be explicitly excluded from the gsplat-only guarantee.

## Pathplanner MassGen Surface

Relevant Pathplanner files:

- `Code/navdp/benchmark/mass_generation.py`
- `Code/navdp/benchmark/example_generation.py`
- `Code/navdp/benchmark/schema.py`
- `Code/navdp/benchmark/cli.py`
- `Code/navdp/benchmark/social_laws.py`
- `Code/navdp/benchmark/validation.py`
- `Code/navdp/benchmark/visualize.py`
- `Code/navdp/data_quality.py`
- `Code/navdp/scene_graph/*`
- `Code/connect_A_B.py`
- `configs/benchmark/human_centric_mission_stream/demo_scene_manifest.json`
- `configs/benchmark/human_centric_mission_stream/human_resources/*`
- `configs/data_quality/scene_exclusions.yaml`
- `scripts/run_mass_generation_platform.sh`
- `scripts/run_mass_generation_container.sh`
- `scripts/run_chingmu_then_interior_mass_generation.sh`
- `scripts/run_chingmu3_platform_smoke.sh`
- `scripts/render_mass_mission_visualizations.py`
- `tests/test_mass_generation.py`
- `tests/test_social_laws.py`
- `tests/test_timeflow_visualization.py`
- `tests/test_human_agent_generation.py`
- `Dockerfile.massgen`
- `Dockerfile.massgen.dockerignore`

Pathplanner schema mission types:

- `navigate_with_social_constraints`
- `deliver_to_human`
- `serve_queue`
- `mission_stream`
- `human_guided_uncertain_region`
- `human_guided_person_disambiguation`
- `human_guided_route_correction`
- `dense_dynamic_avoidance`
- `dense_dynamic_humans`
- `dense_multi_robot`
- `dense_dynamic_combined`

Pathplanner active MassGen mission types:

- `human_guided_uncertain_region`
- `serve_queue`
- `dense_dynamic_humans`
- `dense_dynamic_combined`
- `dense_dynamic_avoidance`
- `dense_multi_robot`
- `mission_stream`
- `deliver_to_human`
- `navigate_with_social_constraints`

Pathplanner active social-navigation MassGen subfamilies:

- `queue_order`
- `personal_space`
- `pedestrian_yield`
- `group_integrity`

Explicitly excluded:

- `functional_space` / L5

Schema-only but not active in MassGen yet:

- `human_guided_person_disambiguation`
- `human_guided_route_correction`

If the request "each new mission family" includes those two schema-only
human-guided mission types, implementation must add builders, validation
coverage, candidate planning, and tests before adding them to
`ACTIVE_MASS_MISSION_TYPES`.

## Platform and CUDA Constraints

MassGen itself is CPU-side scenario construction. It uses NumPy, SciPy, Pillow,
Shapely, scene graphs, wall masks, A*, validation, and JSON/report writes. It
does not render Gaussian splats.

Pathplanner platform docs describe a CPU-only MassGen image:

```text
navdp-massgen:55d9492-cpu1
```

CUDA is needed for Gaussian rendering in `Navdp_Datagen`, not for MassGen
candidate construction.

Local macOS validation should be limited to:

- file/import inspection;
- `py_compile`;
- unit tests that use lightweight builders or do not require local scene data;
- CLI help and candidate-planning smoke tests with temporary manifests;
- Dockerfile/static checks if Docker is available.

CUDA validation must run on the platform:

- perspective render smoke with `GAUSSIAN_RENDER_BACKEND=gsplat`;
- FPV/follow path render smoke;
- any orthographic render verification if the gsplat-only requirement is
  extended to BEV/topdown rendering.

## Implementation Plan

### Phase 1: Make the gsplat rendering contract explicit

1. Add an explicit root dependency on `gsplat` outside the release package.
   Candidate locations:
   - `environment.yml` pip section;
   - a root renderer requirements file, if one is introduced.

2. Add backend visibility to renderer logs.
   - Log once when `_gsplat_render()` is selected.
   - Log or error when the legacy backend is selected.
   - Keep this low-noise in worker processes.

3. Decide strictness.
   - Recommended default: `GAUSSIAN_RENDER_BACKEND=gsplat`.
   - Add an opt-in strict mode such as `STRICT_GSPLAT_RENDER=1` that raises
     instead of falling back to `diff_gaussian_rasterization`.
   - Keep explicit `--backend diff-gaussian` only for debugging legacy parity.

4. Audit root scripts for environment propagation.
   - Add `GAUSSIAN_RENDER_BACKEND="${GAUSSIAN_RENDER_BACKEND:-gsplat}"` in
     shell entry points that launch rendering workers.
   - Ensure parallel render launchers pass the environment through.

5. Resolve the orthographic gap.
   - Option A: implement orthographic support in the `gsplat` wrapper.
   - Option B: document that orthographic debug/BEV rendering remains a legacy
     path and production FPV/follow rendering is gsplat-backed.
   - Option C: in strict mode, fail fast on orthographic Gaussian render until
     Option A is implemented.

### Phase 2: Choose MassGen integration model

Recommended first decision:

- If Pathplanner remains the source of truth, do not port MassGen. Add a small
  documented artifact handoff from Pathplanner scenario outputs to this repo's
  renderer.
- If this repo must own MassGen, port the Pathplanner package wholesale. Avoid a
  partial copy of only `mass_generation.py`; it imports and depends on broader
  `Code.navdp` modules and configs.

If porting into `Navdp_Datagen`, copy these as a coherent unit:

- `Code/connect_A_B.py`
- `Code/navdp/**`
- `configs/benchmark/human_centric_mission_stream/**`
- `configs/data_quality/scene_exclusions.yaml`
- `configs/tasks/**` if benchmark imports or docs require them
- MassGen platform scripts listed above
- MassGen tests listed above
- `Dockerfile.massgen`
- `Dockerfile.massgen.dockerignore`
- a valid MassGen requirements file

Do not copy:

- `__pycache__/`
- `.DS_Store`
- generated `tmp/`
- generated `reports/`
- local symlinks such as `data -> /media/...`
- private `.env` files or API keys

Before copying from Pathplanner, review whether to include the dirty source
edits in `Code/connect_A_B.py` and `Code/navdp/benchmark/example_generation.py`.
They are likely useful performance changes, but they are not committed in the
source repo at inspection time.

### Phase 3: Fix MassGen packaging before platform rebuilds

Pathplanner `Dockerfile.massgen` currently contains:

```text
COPY requirements-massgen.txt /app/requirements-massgen.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-massgen.txt
```

But this checkout did not contain `requirements-massgen.txt`; it only contained
`requirements.txt`.

Before rebuilding the CPU MassGen image, either:

- add `requirements-massgen.txt` with the intended CPU-only subset, or
- change `Dockerfile.massgen` to copy and install `requirements.txt`.

Do this in the repo that will own MassGen.

### Phase 4: Ensure all active mission families are reachable

Required checks after integration:

1. `Code/navdp/benchmark/schema.py` contains every mission type that generated
   scenarios can emit.
2. `Code/navdp/benchmark/mass_generation.py` has the intended active list in
   `ACTIVE_MASS_MISSION_TYPES`.
3. `MASS_GENERATION_GROUP_EASY_TO_HARD` includes every active generation bucket.
   Social-navigation subfamilies are separate buckets using
   `navigate_with_social_constraints:<case_id>`.
4. `Code/navdp/benchmark/cli.py` exposes `--mission-type` and
   `--social-nav-law-case` choices for the active cases.
5. `tests/test_mass_generation.py` asserts:
   - nine active MassGen mission types if the Pathplanner active set is kept;
   - L1-L4 social cases are present;
   - L5 functional space is excluded.
6. Production scripts must match policy:
   - `run_chingmu_then_interior_mass_generation.sh` currently comments out
     `mission_stream` for formal platform runs while it is tuned.
   - `run_chingmu3_platform_smoke.sh` also comments out `mission_stream` in
     `MISSION_TYPE=all`.
   - If "each new mission family" means mission_stream must run in production,
     unpause it only after a platform smoke validates speed and quality.

### Phase 5: Local validation on macOS

Do not run CUDA render tests locally.

Safe local commands after implementation:

```bash
python3 -m py_compile \
  Code/navdp/benchmark/mass_generation.py \
  Code/navdp/benchmark/example_generation.py \
  Code/navdp/benchmark/cli.py \
  Code/connect_A_B.py
```

```bash
python3 -m Code.navdp.benchmark.cli generate-mass-examples --help
```

```bash
python3 -m pytest tests/test_mass_generation.py \
  -k "mass_candidate_planning_includes_nine_types"
```

If tests touch local scene symlinks or unavailable catalogs, patch tests to use
temporary fixtures or the checked-in human-resource fixture paths.

### Phase 6: Platform validation

CPU MassGen platform smoke:

```bash
/app/scripts/run_mass_generation_container.sh preflight
```

Single-scene smoke:

```bash
WORKERS=16 \
TARGET_VALID_PER_TYPE=1 \
MAX_CANDIDATES_PER_TYPE=16 \
MAX_BUILDS=16 \
MISSION_TYPE=all \
/team/telenav/code/Navdp_Datagen_Pathplanner/scripts/run_chingmu3_platform_smoke.sh
```

If `mission_stream` is re-enabled:

```bash
MISSION_TYPE=mission_stream \
WORKERS=8 \
TARGET_VALID_PER_TYPE=1 \
MAX_CANDIDATES_PER_TYPE=16 \
MAX_BUILDS=16 \
/team/telenav/code/Navdp_Datagen_Pathplanner/scripts/run_chingmu3_platform_smoke.sh
```

GPU renderer smoke for gsplat:

```bash
GAUSSIAN_RENDER_BACKEND=gsplat \
python render_label_paths_telesim.py \
  --scenes-dir <scenes-root> \
  --tasks-dir <tasks-root> \
  --scene <scene-id> \
  --label-id <label-id> \
  --output-dir <output-root> \
  --device cuda \
  --video \
  --video-backend nvenc \
  --rgb-frames \
  --save-camera-metadata \
  --overwrite
```

Expected GPU validation result:

- `gsplat` import succeeds;
- no import of `diff_gaussian_rasterization` is required for the perspective
  path;
- a short MP4 and optional RGB frames are written;
- camera metadata is written;
- logs show the selected backend.

## Open Questions

1. Should MassGen remain owned by Pathplanner, with `Navdp_Datagen` only
   rendering its artifacts, or should this repo now own a copy of MassGen?
2. Does the gsplat-only requirement include orthographic Gaussian renders, or
   only production FPV/follow rendering?
3. Should `mission_stream` be re-enabled in formal platform batch scripts now,
   or remain active only through explicit CLI calls until further tuning?
4. Are `human_guided_person_disambiguation` and
   `human_guided_route_correction` required for this pass, or are they still
   schema placeholders?

## Resume Checklist

- [ ] Decide MassGen ownership model.
- [ ] Make root renderer dependency and backend contract explicit.
- [ ] Add strict gsplat fallback behavior or document the allowed legacy cases.
- [ ] If porting MassGen, copy Pathplanner code/config/tests as a coherent unit.
- [ ] Review and decide on Pathplanner dirty performance edits.
- [ ] Fix `Dockerfile.massgen` requirements-file mismatch.
- [ ] Run local non-CUDA validation.
- [ ] Run CPU MassGen smoke on platform.
- [ ] Run GPU gsplat render smoke on platform.
- [ ] Update this document with results and any deviations.
