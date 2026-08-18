# H100 Sustained GPU Utilization Implementation Plan

This is a handoff plan for a future Codex implementation session. The goal is
to make the H100 CPU-encode rendering pipeline sustain high GPU utilization by
reducing setup/load/process overhead and keeping CUDA workers hot.

## Problem Statement

The current H100 strategy is correct at a high level:

- render Gaussian splats on CUDA;
- use CPU `libx264` for video because H100 has no NVENC video encode block;
- run multiple independent render workers per GPU;
- use `--actor-gpu-resident` when human actors are present.

The weakness is structural. The current pipeline can produce short bursts of
near-100% GPU utilization, but average utilization drops because many worker
seconds are spent outside the CUDA render loop.

The 2026-08-18 weekly report captured this clearly:

```text
out/weekly_progress_report_20260818/
```

Relevant report artifacts:

```text
out/weekly_progress_report_20260818/WEEKLY_PROGRESS_REPORT.md
out/weekly_progress_report_20260818/metrics/weekly_summary_metrics.json
out/weekly_progress_report_20260818/metrics/gpu_usage_vram_nvml_10hz_summary.json
out/weekly_progress_report_20260818/metrics/smoketest_benchmarks/results_workersweep_cpu_w32_10hz_nvml_benchmark_summary.json
out/weekly_progress_report_20260818/assets/graphs/gpu_usage_vram_nvml_10hz.png
out/weekly_progress_report_20260818/assets/graphs/worker_sweep_gpu_latency.png
```

Key evidence from the worker=32 CPU-encode smoke:

- average GPU utilization: about 56%;
- p90 / p95 GPU utilization: 99% / 100%;
- samples at or above 80% GPU utilization: about 35%;
- peak VRAM: about 27.5 GiB on RTX 5880-class hardware;
- top-level selected records: 32;
- nested renderer paths: 160;
- nested renderer frames: 8,384;
- summed top-level render wall time: about 1,976 seconds;
- summed nested renderer-loop time: about 312 seconds;
- CPU H.264 encode time inside nested renderer metrics: about 21.8 seconds,
  roughly 7% of nested renderer time.

Interpretation: CPU encode is not the main cause of the utilization drops in
that run. The dominant loss is worker lifetime spent in process launch, Python
import, scene load, renderer initialization, actor setup, serial per-job
execution, output bookkeeping, and tail gaps.

## Current Architecture Hotspots

Current relevant files:

```text
scripts/massgen/run_family_rollout_h100.py
scripts/massgen/run_render_smoketest_benchmark.py
scripts/massgen/render_manifest_jobs.py
navdp_datagen/massgen/render_executor.py
render_label_paths_telesim.py
gaussian_renderer/__init__.py
utils/telesim_actor_utils.py
utils/video_writer_utils.py
```

Observed structural issues:

1. `scripts/massgen/run_family_rollout_h100.py` fans out families across worker
   slots, pins each child process to one GPU, assigns CPU thread budgets, and
   monitors GPU utilization. This is a good outer orchestration layer, but it
   does not make individual workers persistent.

2. `scripts/massgen/render_manifest_jobs.py` builds a plan and, with
   `--execute`, calls `execute_render_plans`.

3. `navdp_datagen/massgen/render_executor.py::execute_render_plans` executes
   planned render commands sequentially within a manifest. A manifest with five
   robot-view jobs therefore invokes five renderer commands in sequence.

4. Each planned command runs `render_label_paths_telesim.py` as a fresh Python
   process. That process performs imports, prechecks, scene metadata loading,
   Gaussian renderer construction, scene PLY loading, actor loading, optional
   actor GPU cache upload, then enters the actual per-frame render loop.

5. `render_label_paths_telesim.py` records detailed per-path stage timings, but
   `duration_total_sec` mostly covers the hot path after scene setup. It does
   not explain the full outer `render_elapsed_sec`.

6. `gaussian_renderer/__init__.py` currently calls gsplat `rasterization()` with
   one camera at a time: `viewmats[None]`, `Ks[None]`, and `packed=False`.
   Local gsplat supports batched camera tensors, but this wrapper does not use
   them yet.

## Target Outcome

For H100-class runs, target these end-to-end metrics:

- average GPU utilization: at least 80%;
- samples at or above 80% GPU utilization: at least 70%;
- throughput improves versus the current multi-process fanout baseline;
- no unacceptable increase in CUDA OOM, segmentation fault, corrupt output, or
  retry rate;
- CPU encode remains bounded and does not starve the render queue;
- per-worker VRAM use remains predictable under configured budgets.

Utilization alone is not enough. A change that raises utilization while reducing
end-to-end frames/sec should not be accepted unless it is a diagnostic-only
experiment.

## Implementation Phases

### Phase 1: Make Missing Time Visible

Add timing around the currently hidden setup and process phases.

Implementation status, 2026-08-18:

- `render_label_paths_telesim.py` emits a top-level `lifecycle_seconds` object
  beside existing `stage_seconds` path metrics.
- `scripts/massgen/run_render_smoketest_benchmark.py` now summarizes hot
  render-loop time, renderer process totals, hidden overhead, setup seconds per
  frame, and process launches per 1,000 rendered frames.
- Existing `stage_seconds` keys remain unchanged for older report scripts.
- Next validation step: run the same smoke package on 5880/H100-class hardware
  and compare `render_overhead_summary` against GPU utilization timelines.

Suggested new metrics:

```text
process_start_sec
python_import_sec
manifest_plan_sec
precheck_sec
scene_metadata_load_sec
scene_asset_build_sec
renderer_init_sec
scene_ply_load_sec
actor_plan_load_sec
actor_sequence_load_sec
actor_gpu_cache_upload_sec
first_frame_latency_sec
render_loop_sec
writer_close_sec
output_bookkeeping_sec
process_total_sec
```

Implementation guidance:

- Keep existing `stage_seconds` keys stable for current reports.
- Add a new `lifecycle_seconds` object rather than overloading per-frame stage
  metrics.
- Record both per-renderer-process metrics and aggregate benchmark summaries.
- In `run_render_smoketest_benchmark.py`, summarize:
  - outer render wall time;
  - nested renderer-loop time;
  - hidden overhead time;
  - overhead percentage;
  - process launches per 1,000 frames;
  - setup seconds per frame.

Acceptance criteria:

- The worker=32 weekly-report result can be explained without manual `jq`.
- For each top-level record:
  `outer_render_elapsed ~= lifecycle/process_total + known subprocess overhead`.
- Reports include "hot render loop vs setup/load/process overhead".

### Phase 2: Group Same-Scene Jobs Into Fewer Renderer Invocations

Before building persistent workers, reduce the number of renderer launches.

Design:

- In `render_executor.py`, group executable plans by compatible renderer state:
  - same scene root;
  - same scene id;
  - same Gaussian model path;
  - same resolution;
  - same FOV / znear / zfar;
  - same video backend;
  - same depth/RGB/camera metadata requirements;
  - compatible actor plan handling.
- For compatible camera-only or simple actor jobs, invoke
  `render_label_paths_telesim.py` once with multiple `--label-id` values.
- Keep separate invocations when robot overlay dependencies or incompatible
  actor plans make grouping unsafe.

Important constraint:

- Current MassGen jobs often use different actor plans per robot-view job. Group
  only when the renderer can load the correct actor plan for each label. If that
  is not true yet, implement grouped camera-only/static cases first and leave
  actor-plan grouping for Phase 3.

Expected win:

- fewer Python process launches;
- fewer scene PLY loads;
- fewer renderer initializations;
- less low-util warmup time.

Acceptance criteria:

- Existing one-job manifests produce byte-valid videos and metrics.
- Multi-job same-scene manifests reduce renderer process count.
- Output paths, metrics JSON paths, actor metadata, depth maps, and robot overlay
  inputs remain compatible with existing audit scripts.

### Phase 3: Persistent Per-GPU Worker Pool

This is the main structural fix.

Design:

- Add a coordinator process that owns the global work queue.
- Start long-lived worker processes.
- Pin each worker to:
  - one physical GPU via `CUDA_VISIBLE_DEVICES`;
  - a CPU core set via `taskset` where available;
  - bounded CPU library thread counts.
- Each worker receives work items over an IPC queue.
- A work item should represent a renderable unit:
  - scene id;
  - Gaussian model path;
  - sensor/render options;
  - label path(s);
  - actor plan path(s);
  - output root;
  - metadata/depth/video options.
- Worker keeps loaded scene and renderer state across multiple work items.
- Worker reports metrics back to coordinator after every output unit.
- Coordinator writes JSONL records incrementally.

Worker cache policy:

- Keep current scene loaded while next work item uses the same scene.
- Cache N most recent scenes only if VRAM budget allows.
- Keep actor sequences GPU-resident when full sequence fits.
- Use all-or-nothing actor GPU caching; avoid partial caches unless measured.
- Evict idle actor sequences before evicting active scene state.

Failure policy:

- If a worker hits CUDA OOM:
  - evict idle actor caches;
  - retry once;
  - if still failing, restart the worker process and mark the work item retryable.
- If a worker segfaults:
  - coordinator detects process exit;
  - restarts the worker;
  - requeues in-flight work once;
  - records failure after configured retry count.
- Keep output writes idempotent and resume-safe.

Expected win:

- scene/actor setup is amortized over many labels;
- workers spend most of lifetime in render or encode/output;
- GPU sees fewer long idle valleys.

Acceptance criteria:

- On the same benchmark package, process launches per 1,000 frames drop sharply.
- Hidden overhead percentage drops below 25% before moving to batching.
- GPU utilization samples above 80% increase materially without worse throughput.

### Phase 4: Keep The Queue Deep And Avoid Tail-Off

The scheduler must not run exactly one wave of work and then drain slowly.

Design:

- Sort work by expected cost before dispatch:
  - longer frame count first;
  - heavier actor count first;
  - same-scene groups together;
  - same actor resources together.
- Maintain more queued work than active workers.
- Use dynamic refill: as soon as a worker finishes, dispatch the next compatible
  work item.
- Avoid putting several short jobs at the end of the queue where most GPUs have
  already gone idle.

Acceptance criteria:

- GPU utilization does not collapse during the final 20% of a long benchmark
  unless the global queue is actually exhausted.
- Per-worker completed-frame counts are reasonably balanced.

### Phase 5: Camera/Viewpoint Batching In gsplat

Once workers are persistent, add intra-process batching.

Local gsplat supports batched rasterization by passing batched `viewmats` and
`Ks`. Current wrapper uses one camera per call.

Batch candidates, from safest to hardest:

1. camera-only/static scene labels;
2. multiple robot viewpoints at the same timestamp with identical actor state;
3. consecutive frames where active actor count and Gaussian tensor shapes are
   stable;
4. labels padded to a fixed Gaussian count using zero-opacity actors.

Benchmark matrix:

```text
batch_size = 1, 2, 4, 8
packed = false, true
render_mode = RGB, RGB+D
resolution = production target and smoke target
actor_count = 0, 1, 4, 8
```

Implementation notes:

- Do not batch across incompatible scene Gaussian counts unless using padding.
- Watch VRAM; `packed=False` is faster but can consume much more memory.
- Preserve current single-frame path as fallback.
- Validate image equality or near-equality against the existing path.

Acceptance criteria:

- Batch size 2 or 4 improves frames/sec on H100 without unacceptable VRAM growth.
- Batched outputs pass visual spot checks and per-frame metadata alignment.

### Phase 6: Async CPU Encode / Output Pipeline

H100 requires CPU video encode, but encode should not block CUDA submission.

Design:

- Render worker produces GPU tensors.
- Convert RGB to uint8 on GPU.
- Copy to pinned host buffers asynchronously where possible.
- Push host frames into a bounded encode queue.
- CPU encode thread/process owns `imageio`/FFmpeg writer.
- Render loop continues while encode drains, up to the queue limit.
- Backpressure when the queue is full.

Implementation notes:

- Keep output order exact.
- Keep queue bounded to avoid RAM blowups.
- Include queue wait time in metrics.
- Use local NVMe/scratch for hot writes where possible.

Acceptance criteria:

- CPU encode wait time is visible and bounded.
- Render loop no longer blocks on every `writer.append_data` when CPU encode is
  slower than CUDA rendering.
- End-to-end frames/sec improves or remains stable while utilization improves.

## Benchmark Plan

Use three benchmark tiers.

### Tier 1: Short Functional Smoke

Purpose: catch correctness regressions quickly.

Example:

```text
minimal_frames = 16 or 24
families = 1 to 2
workers = 1 to 4
```

Pass criteria:

- all expected videos exist;
- metrics JSON is valid;
- camera/actor metadata exists when requested;
- no obvious visual failures.

### Tier 2: Setup-Amortization Benchmark

Purpose: measure overhead reduction.

Example:

```text
minimal_frames = 64
records = 32
workers = existing baseline and new persistent workers
```

Compare:

- process launches;
- hidden overhead percentage;
- total frames/sec;
- average GPU utilization;
- samples at or above 80%.

### Tier 3: H100 Sustained Run

Purpose: determine whether the design reaches sustained target.

Example:

```text
minimal_frames = disabled or high enough to amortize setup
jobs_per_gpu = 4, 6, 8
duration = at least several minutes
video_backend = cpu
actor_gpu_resident = enabled
```

Pass criteria:

- average GPU utilization >= 80%;
- at least 70% of samples >= 80%;
- throughput improves versus current H100 runner;
- no persistent OOM/retry instability.

## Metrics To Preserve

Do not remove existing metric keys consumed by current reports:

```text
actor_visibility_sec
actor_transform_sec
actor_tensor_pack_sec
actor_merge_update_sec
gaussian_render_sec
gpu_readback_sec
perframe_light_sec
camera_metadata_sec
perframe_depth_sec
perframe_png_sec
mp4_write_sec
h264_encode_sec
h264_mux_sec
video_close_sec
render
encode
measured_total_sec
```

Add new lifecycle metrics beside them.

## Risks

- Persistent CUDA workers can retain bad state after exceptions. Use worker
  restart on OOM/segfault rather than trying to recover every failure in-process.
- Grouping actor-plan jobs is subtle. Keep conservative compatibility checks.
- Batching moving actors may break metadata alignment or visual correctness if
  frame order is mishandled.
- `packed=False` batching can exhaust VRAM. H100 has more headroom, but it is
  not infinite.
- Async encode can reorder frames unless the queue protocol is strict.
- Multi-process GPU scheduling may still have context overhead. If utilization
  remains sawtooth after persistent workers, benchmark NVIDIA MPS as a separate
  controlled experiment.

## Suggested Ticket Breakdown

1. Add lifecycle timing and overhead summaries.
2. Add renderer process-count reporting to smoke and family rollout summaries.
3. Implement conservative same-scene renderer invocation grouping.
4. Add persistent worker prototype for camera-only/static jobs.
5. Extend persistent worker to actor-plan jobs.
6. Add worker cache eviction and restart policy.
7. Add gsplat batched-camera render path behind a flag.
8. Add async CPU encode/output queue behind a flag.
9. Run H100 job-per-GPU sweep and document accepted defaults.

## Definition Of Done

This effort is done when an H100 run can sustain the target on production-like
work, not only on synthetic bursts:

```text
avg_gpu_util_pct >= 80
samples_ge80_pct >= 70
throughput_fps > current H100 multi-process baseline
hidden_overhead_pct <= 25
retry/OOM/segfault rate acceptable for unattended overnight runs
```
