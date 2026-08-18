# Weekly Progress Report Runbook

This document records the process used to assemble
`out/weekly_progress_report_20260818/` so future report runs can follow the
same structure and validation standard.

The weekly bundle is a curated artifact, not a raw test dump. Raw renderer
outputs can be large and unstable; copy only the report-worthy videos, stills,
graphs, tables, logs, metrics, and raw references into the final bundle, then
remove temporary test roots.

## Output Contract

Use a dated root under ignored `out/`:

```text
out/weekly_progress_report_<YYYYMMDD>/
  WEEKLY_PROGRESS_REPORT.md
  CLEANUP_NOTES.md
  assets/
    videos/
    frames/
    graphs/
    tables/
  logs/
  metrics/
  raw_refs/
```

Required final files:

- `WEEKLY_PROGRESS_REPORT.md`: executive summary, visuals table, measurements,
  implementation changes, validation status, known issues.
- `CLEANUP_NOTES.md`: exact local and remote temporary paths removed, plus
  cleanup scope.
- `assets/videos/`: representative MP4s only. Do not include videos with no
  visible target actor/robot as positive evidence.
- `assets/graphs/`: contact sheets, GPU/VRAM plots, depth/foreground
  comparisons, worker sweep plots.
- `assets/tables/`: CSV and markdown versions of measured summaries.
- `logs/`: compact command logs for the runs cited in the report.
- `metrics/`: renderer metrics JSON, benchmark summaries, family rollout
  summaries.
- `raw_refs/`: copied source notes/config/audit files that explain how results
  were produced.

The 2026-08-18 bundle is the reference example:

```text
out/weekly_progress_report_20260818/
```

## Preflight

Record these before running anything:

- local commit and branch;
- remote host, checkout path, Python env, and GPU;
- remote output root;
- source data roots for scenes, human GS assets, robot GLB/URDF, and Kimodo
  JSONs;
- exact render backend and video backend;
- whether the run is 5880/NVENC-oriented or H100/CPU-encode-oriented.

Example 5880 config fields are preserved in:

```text
out/weekly_progress_report_20260818/raw_refs/run_config_5880.json
```

Key 2026-08-18 remote paths:

```text
host: 5880 / codex-5880
repo: /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting
DATA scene/assets root: /mnt/DATA/dongjk/navdp_data
DATA1 test root: /mnt/DATA1/dongjk/navdp_data
human GS root: /mnt/DATA/dongjk/navdp_data/human_gs_source
Kimodo SMPL-X JSON root: /mnt/DATA/dongjk/navdp_data/assets/walking_kimodo
```

Use unique remote roots per report run so cleanup is unambiguous:

```text
/mnt/DATA1/dongjk/navdp_data/<run_name>/
```

## Mission-Family Rollout

Purpose: produce one reviewable MP4 plus planner BEV PNG/GIF for each mission
family.

Inputs:

- packaged planned examples;
- scene assets copied or referenced on 5880;
- human GS source override with valid local human IDs;
- renderer checkout matching the branch under test.

Representative 2026-08-18 command:

```sh
cd /tmp/navdp_datagen_massgen_test
/home/lenovo/miniconda3/envs/telesim3d_kuro/bin/python scripts/massgen/run_family_rollout.py \
  --package-root /tmp/massgen_family_rollout_clearout_rerun/package \
  --results-root /tmp/massgen_family_rollout_clearout_rerun/results \
  --python-bin /home/lenovo/miniconda3/envs/telesim3d_kuro/bin/python \
  --video-backend cpu \
  --device cuda \
  --limit 1 \
  --retry 3
```

If one family fails from intermittent actor PLY loading, rerun only that family
with a higher retry count, then merge it back into the result tree. The
2026-08-18 `group_integrity` run needed this.

Validation gates:

- every family has at least one MP4;
- every family has planner BEV PNG and GIF sidecars;
- camera metadata exists;
- label path and camera path agree within a small XY tolerance;
- visually inspect queue/group/human-facing cases;
- do not accept a video with invisible or missing required actors.

Run the audit helper and preserve its JSON:

```sh
python scripts/massgen/audit_family_rollout_results.py \
  --results-root <LOCAL_DOWNLOADED_RESULTS> \
  --output-json <WEEKLY_ROOT>/raw_refs/family_rollout_audit.json
```

The 2026-08-18 audit output is:

```text
out/weekly_progress_report_20260818/raw_refs/family_rollout_audit.json
```

## Render Smoke And Throughput Benchmarks

Purpose: measure renderer speed, storage, GPU/VRAM usage, CPU encode behavior,
and worker scaling.

Build or reuse a formal smoke package. The 2026-08-18 package had:

- `1,800` entries;
- `36` family/source/scene groups;
- `32` unique source scenes;
- `1,400` currently renderable simple-renderer entries;
- `400` expected blocked multi-robot entries;
- `2,200` renderable camera jobs because dense avoidance has multiple camera
  jobs per scenario.

Run templates are stored in:

```text
out/weekly_progress_report_20260818/raw_refs/run_config_5880.json
```

CPU-encode template for H100 simulation on 5880:

```sh
cd /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting
conda run --no-capture-output -n cuda121 python scripts/massgen/run_render_smoketest_benchmark.py \
  --package-root <PACKAGE_ROOT> \
  --results-root <RESULTS_ROOT> \
  --repo-root /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting \
  --python-bin /home/dongjk/.conda/envs/cuda121/bin/python \
  --render-script /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting/render_label_paths_telesim_massgen.py \
  --video-backend cpu \
  --minimal-frames 32 \
  --workers <WORKERS> \
  --gpu-sample-interval-sec 0.25 \
  --clean
```

NVENC template for 5880:

```sh
cd /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting
IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg \
conda run --no-capture-output -n cuda121 python scripts/massgen/run_render_smoketest_benchmark.py \
  --package-root <PACKAGE_ROOT> \
  --results-root <RESULTS_ROOT> \
  --repo-root /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting \
  --python-bin /home/dongjk/.conda/envs/cuda121/bin/python \
  --render-script /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting/render_label_paths_telesim_massgen.py \
  --video-backend nvenc \
  --minimal-frames 32 \
  --workers <WORKERS> \
  --gpu-sample-interval-sec 0.25 \
  --clean
```

Worker sweep:

```text
workers: 1, 2, 4, 8, 16, 32
minimal frame cap: 32 or 64
GPU sampling: NVML timeline at 10 Hz when collecting final report metrics
```

Acceptance metrics:

- record success count;
- actual MP4 count;
- rendered frame count from successful records;
- total wall time;
- average, p50, p90, p95, and max GPU utilization;
- percentage of samples with GPU utilization >=80%;
- max VRAM;
- output size.

Do not use peak GPU utilization alone. The 2026-08-18 run hit 100% peaks but
only kept `35.2%` of samples above 80% in the 32-worker run.

## Depth And Robot Overlay Validation

Purpose: prove metric depth encoding, depth-gated GLB composition, and G1/Kimodo
animation.

Required comparisons:

- foreground overlay video or frames;
- depth-composited video or frames;
- still-frame comparison contact sheet;
- renderer metrics for no-depth and depth-map runs;
- overlay logs for foreground and depth passes.

Use the shared metric depth contract:

- renderer writes absolute metric depth PNGs;
- GLB compositor decodes the same quantization;
- depth composition must allow GS walls/objects to occlude GLB robots.

For G1/Kimodo probes, use a camera that is deliberately generated relative to
the robot pose. A video is not accepted unless the robot is visible in sampled
frames. The later `out/remote_validation_5880/90e88da_g1_smoke/49_g1_robot.mp4`
run is an example of what not to accept: the script completed, but the robot was
not visible because the fallback placement used camera-path tangent placement
instead of guaranteed in-frame robot-follow placement.

Positive 2026-08-18 references:

```text
out/weekly_progress_report_20260818/assets/videos/depth_composited_multi_robot_24f.mp4
out/weekly_progress_report_20260818/assets/videos/foreground_multi_robot_old_comparison.mp4
out/weekly_progress_report_20260818/assets/videos/g1_kimodo_armfix_follow_camera.mp4
out/weekly_progress_report_20260818/assets/graphs/foreground_vs_depth_frame_comparison.png
out/weekly_progress_report_20260818/assets/graphs/g1_kimodo_armfix_contact_sheet.png
```

Quality gates:

- sample frames from each video;
- confirm expected robot/human pixels are visible;
- confirm depth and foreground differ where occlusion should happen;
- reject empty-overlay videos even if the MP4 exists;
- keep logs and frame contact sheets.

## Report Assembly

The final report should have these sections:

1. Executive summary.
2. Representative visuals table.
3. Depth-composited robot placement and measured overhead.
4. Mission-family rollout examples table.
5. GPU/throughput smoke-test metrics.
6. Time and space estimates.
7. Implementation changes.
8. Validation.
9. Cleanup performed.
10. Known issues / next work.

The `WEEKLY_PROGRESS_REPORT.md` should link to relative files inside the
bundle. Keep generated videos/graphs/tables in `assets/`; keep raw command
notes and config in `raw_refs/`.

When adding tables, store both machine-readable and report-readable versions:

```text
assets/tables/<name>.csv
assets/tables/<name>.md
```

For plots and contact sheets, use stable filenames that describe the metric or
comparison:

```text
assets/graphs/gpu_usage_vram_nvml_10hz.png
assets/graphs/worker_sweep_gpu_latency.png
assets/graphs/family_video_contact_sheet.png
assets/graphs/foreground_vs_depth_frame_comparison.png
```

## Cleanup

After copying curated artifacts into the weekly bundle:

1. Remove local raw outputs that were created only for the run.
2. Remove remote temporary roots created under the run-specific DATA1 folder.
3. Do not remove dataset roots, scene roots, human assets, source repos, or
   package roots that were not created for this run.
4. Record exact removed paths in `CLEANUP_NOTES.md`.
5. Verify remote free space after cleanup.

The 2026-08-18 cleanup reference is:

```text
out/weekly_progress_report_20260818/CLEANUP_NOTES.md
```

## Known Operational Issues

- 5880 renderer imports and `pyrender` subprocesses can intermittently segfault.
  Retry can recover, but persistent workers are the proper long-term fix.
- System ffmpeg with CPU backend showed native instability in smoke runs; for
  CPU/H100 simulation prefer ImageIO bundled ffmpeg unless the writer is
  replaced.
- H100 has no NVENC. Keep H100 CPU-encode pipeline separate from 5880 NVENC
  assumptions.
- CHINGMU scenes should use `3dgs_raw.ply`; some `3dgs_compressed.ply` files
  are packed and lack direct `x/y/z` fields for this renderer.
- Wall masks in selected source scenes were missing or placeholder-like in the
  2026-08-18 smoke package. This does not block rendering but does matter for
  planner-side path filtering and path-quality ranking.
- Do not accept command success as visual success. For actor/robot work, require
  visible-pixel or sampled-frame evidence.
