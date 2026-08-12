# Mission-Family Rollout Handoff

Date: 2026-08-07
Branch: `massgen`

Use this as the living handoff while updating rendering one mission family at a
time. Keep it current after each implementation pass, 5880 smoke test, artifact
download, and cleanup.

## Scope

The rollout order stays easy-to-hard:

1. `deliver_to_human`
2. `serve_queue`
3. `human_guided_uncertain_region`
4. `navigate_with_social_constraints:personal_space`
5. `navigate_with_social_constraints:queue_order`
6. `navigate_with_social_constraints:group_integrity`
7. `navigate_with_social_constraints:pedestrian_yield`
8. `dense_dynamic_humans`
9. `dense_dynamic_avoidance`
10. `dense_multi_robot`
11. `dense_dynamic_combined`
12. `mission_stream`

Schema-only Pathplanner families remain out of renderer scope until they have
active builders and examples:

- `human_guided_person_disambiguation`
- `human_guided_route_correction`

## Current State

Renderer repo:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen
```

Pathplanner repo:

```text
/Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner
```

5880 render checkout for clean MassGen work:

```text
/tmp/navdp_datagen_massgen_benchmark_20260807
```

Avoid using this dirty 5880 checkout for branch switching until its submodules
are handled:

```text
/home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting
```

Known dirty submodules there:

- `SIBR_viewers`
- `submodules/diff-gaussian-rasterization_custom`
- `submodules/fused-ssim`
- `submodules/simple-knn`

## Per-Family Workflow

For each mission family, do the same loop.

1. Generate or copy five Pathplanner scenario JSONs into:

   ```text
   tests/inputs/massgen_generated/<family_or_subfamily>/
   ```

2. Convert or prepare a render manifest with strict and non-strict preflight:

   ```bash
   scripts/massgen/prepare_render_run.py \
     --config-json <config.json> \
     --write \
     --summary
   ```

3. Plan renderer jobs before running CUDA:

   ```bash
   scripts/massgen/render_manifest_jobs.py \
     --manifest-json <render_manifest.json> \
     --family <family_or_subfamily> \
     --write-inputs \
     --json
   ```

4. Add or update local tests for the family-specific behavior.

5. Run local non-CUDA checks:

   ```bash
   python3 -m py_compile \
     navdp_datagen/massgen/render_executor.py \
     scripts/massgen/render_manifest_jobs.py

   /Users/dongjk/miniconda3/bin/python3.13 -m pytest \
     tests/test_massgen_render_executor.py \
     tests/test_massgen_render_run_config.py \
     tests/test_massgen_render_manifest.py
   ```

6. Commit and push before 5880 testing.

7. Pull the clean 5880 checkout:

   ```bash
   ssh 5880host 'cd /tmp/navdp_datagen_massgen_benchmark_20260807 && \
     git fetch origin massgen && \
     git checkout -B massgen origin/massgen && \
     git status --short --branch && \
     git log --oneline -1'
   ```

8. Run one smoke render on 5880 with `GAUSSIAN_RENDER_BACKEND=gsplat`.

9. Download the representative outputs for visual review.

10. Record pass/fail, command, commit, output paths, downloaded files, and cleanup
    status in the family table below.

11. Clear temporary local and 5880 files after review.

## 5880 Smoke-Test Template

Use a family-specific output root under `/mnt/DATA` so cleanup is unambiguous:

```text
/mnt/DATA/dongjk/navdp_data/outputs/massgen_family_smoke/<family>/<commit_or_date>/
```

Basic remote command shape:

```bash
ssh 5880host '
  set -e
  cd /tmp/navdp_datagen_massgen_benchmark_20260807
  export GAUSSIAN_RENDER_BACKEND=gsplat
  scripts/massgen/render_manifest_jobs.py \
    --manifest-json <remote_render_manifest.json> \
    --family <family_or_subfamily> \
    --output-root /mnt/DATA/dongjk/navdp_data/outputs/massgen_family_smoke/<family>/<commit_or_date> \
    --write-inputs \
    --execute \
    --video-backend nvenc \
    --limit 1
'
```

Expected evidence to collect:

- command used;
- git commit on 5880;
- scenario id and scene id;
- selected job id and robot id;
- MP4 path;
- camera metadata path;
- metrics JSON path;
- stdout/stderr log path if captured;
- whether `GAUSSIAN_RENDER_BACKEND=gsplat` was active;
- whether target humans, queue actors, moving humans, or peer robots are visible
  as expected.

## Artifact Download Template

Download only the files needed for visual review:

```bash
mkdir -p out/massgen_family_smoke/<family>/<commit_or_date>
rsync -av \
  5880host:/mnt/DATA/dongjk/navdp_data/outputs/massgen_family_smoke/<family>/<commit_or_date>/renders/ \
  out/massgen_family_smoke/<family>/<commit_or_date>/renders/
rsync -av \
  5880host:/mnt/DATA/dongjk/navdp_data/outputs/massgen_family_smoke/<family>/<commit_or_date>/metrics/ \
  out/massgen_family_smoke/<family>/<commit_or_date>/metrics/
```

Local smoke artifacts should stay under ignored paths such as `out/` or
`benchmark_outputs/`. Do not commit MP4s, PNGs, metrics dumps, generated labels,
or `.DS_Store` files.

## Cleanup Template

Before deleting, print the exact path and list a few files:

```bash
ssh 5880host '
  root=/mnt/DATA/dongjk/navdp_data/outputs/massgen_family_smoke/<family>/<commit_or_date>
  printf "cleanup root: %s\n" "$root"
  find "$root" -maxdepth 3 -type f | sort | head -50
'
```

After visual review, remove only that one family smoke root:

```bash
ssh 5880host '
  root=/mnt/DATA/dongjk/navdp_data/outputs/massgen_family_smoke/<family>/<commit_or_date>
  test -n "$root" && test -d "$root" && find "$root" -depth -delete
'
```

Local cleanup:

```bash
root=out/massgen_family_smoke/<family>/<commit_or_date>
test -n "$root" && test -d "$root" && find "$root" -depth -delete
```

After cleanup, verify:

```bash
git status --short --branch
git ls-files --others --exclude-standard
```

## Rollout Progress

| Step | Family | Status | Local Evidence | 5880 Evidence | Downloaded Artifacts | Cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `deliver_to_human` | Local human-only path ready | Actor-bundle materialization added. Planned command passes `--actor-plan-json`, visibility culling, actor metadata, and `GAUSSIAN_RENDER_BACKEND=gsplat`. Local tests pass. | Not run after actor-bundle work. | None yet. | None yet. |
| 2 | `serve_queue` | Local human-only path ready | One-human and multi-human queue manifests use actor bundles with `queue_wait`; per-frame actor metadata records visible/culled/rendered actors. | Not run. | None. | None. |
| 3 | `human_guided_uncertain_region` | Local human-only path ready | Informant + one robot is covered by actor-bundle tests with `wave` action selection. | Not run. | None. | None. |
| 4 | `navigate_with_social_constraints:personal_space` | Local human-only path ready | One human + one robot is covered by social-law family selection and actor-bundle tests. | Not run. | None. | None. |
| 5 | `navigate_with_social_constraints:queue_order` | Local human-only path ready | One-human and multi-human queue-order manifests use actor bundles with `queue_wait`. | Not run. | None. | None. |
| 6 | `navigate_with_social_constraints:group_integrity` | Local human-only path ready | One-human group-integrity social-law selection maps to `stand`; multi-human grouping can use the same actor-bundle renderer if assets are present. | Not run. | None. | None. |
| 7 | `navigate_with_social_constraints:pedestrian_yield` | Local human-only path ready | One moving human + one robot is covered by actor-plan interpolation tests with `walk` action selection. | Not run. | None. | None. |
| 8 | `dense_dynamic_humans` | Local human-only path ready | One-human and multi-human moving-human manifests use actor bundles; culling counters are emitted per actor/frame. | Not run. | None. | None. |
| 9 | `dense_dynamic_avoidance` | Local one-robot/human-only path ready | One-robot moving-human jobs use actor bundles; jobs with peer robots still block on GLB overlay integration. | Not run. | None. | None. |
| 10 | `dense_multi_robot` | Not started | Requires peer-robot GLB overlay integration. | Not run. | None. | None. |
| 11 | `dense_dynamic_combined` | Not started | Requires moving humans plus peer robots. | Not run. | None. | None. |
| 12 | `mission_stream` | Not started | Requires multi-robot viewpoint support and child mission metadata. | Not run. | None. | None. |

## Current `deliver_to_human` Todo

- [x] Add manifest-driven executor planning.
- [x] Select jobs by family, job id, robot id, and sensor.
- [x] Materialize robot FPV trajectories into TeleSim label-path JSONs.
- [x] Force planned commands to use `GAUSSIAN_RENDER_BACKEND=gsplat`.
- [x] Add local tests for `deliver_to_human` command planning.
- [x] Connect human-only manifest Gaussian composition through a generated
  actor-bundle `actor_plans/<job_id>.json` and
  `render_label_paths_telesim.py --actor-plan-json`.
- [x] Add local coverage for human-only `deliver_to_human`, `serve_queue`,
  `human_guided_uncertain_region`,
  `navigate_with_social_constraints:personal_space`,
  `navigate_with_social_constraints:queue_order`,
  `navigate_with_social_constraints:group_integrity`,
  `navigate_with_social_constraints:pedestrian_yield`,
  `dense_dynamic_humans`, and one-robot `dense_dynamic_avoidance`.
- [x] Add multi-human human-only actor-bundle coverage for `serve_queue`,
  `navigate_with_social_constraints:queue_order`, and `dense_dynamic_humans`.
- [x] Add baseline-vs-optimized human-only actor benchmark wrapper:
  `scripts/massgen/benchmark_simple_actor_render.py`.
- [ ] Generate/copy five real `deliver_to_human` scenarios under
  `tests/inputs/massgen_generated/deliver_to_human/`.
- [ ] Ensure strict preflight blocks missing target-human action assets before GPU
  work starts.
- [ ] Run one 5880 smoke render where the target human is visible.
- [ ] Download the smoke MP4 and metrics for visual review.
- [ ] Clean local and 5880 smoke artifacts after review.

## Human-Only Actor-Bundle Boundary

The current executor bridges MassGen jobs into the existing
`render_label_paths_telesim.py` label-path renderer. Jobs with manifest humans
and no peer robots now generate:

- `label_paths/<job_id>.json` for the robot FPV path;
- `actor_plans/<job_id>.json` as a `massgen_actor_bundle.v1` with one entry per
  manifest human;
- renderer commands with `--actor-plan-json`, `--save-actor-metadata`, and
  optional `--actor-gpu-resident`.

The renderer keeps the camera on the robot path and places every actor from the
manifest plan bundle. It writes `<label>_actors.json` with per-frame
candidate, visible, culled, and rendered actor records.

The executor still blocks jobs with peer robots and humans that use multiple
distinct PLY action sequences in one job. Peer robots require the GLB overlay
path before `dense_multi_robot`, `mission_stream`, and peer-robot variants of
`dense_dynamic_avoidance` / `dense_dynamic_combined` can be considered complete.

## Baseline/Optimized Actor Benchmark

After a human-only manifest passes local planning, run the paired benchmark on
5880 before and after multi-target optimization work:

```bash
ssh 5880host '
  set -e
  cd /tmp/navdp_datagen_massgen_benchmark_20260807
  export GAUSSIAN_RENDER_BACKEND=gsplat
  scripts/massgen/benchmark_simple_actor_render.py \
    --manifest-json <remote_render_manifest.json> \
    --family deliver_to_human \
    --output-root /mnt/DATA/dongjk/navdp_data/outputs/massgen_family_smoke/actor_benchmark/<commit_or_date> \
    --video-backend nvenc \
    --minimal-frames 120 \
    --limit 1
'
```

The helper runs `baseline_cpu_actor` and `optimized_gpu_actor_cache` against the
same selected job and writes `simple_actor_benchmark_summary.json` plus
`simple_actor_benchmark_report.md`. Compare wall time and actor stages
(`actor_transform_sec`, `actor_tensor_pack_sec`, `actor_merge_update_sec`, and
`actor_gpu_cache_upload_sec`) before changing the multi-target renderer.
