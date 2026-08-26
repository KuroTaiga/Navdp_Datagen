# H100 Persistent Pipeline Session Handoff

Date: 2026-08-21  
Branch: `massgen`  
Latest pushed commit at handoff start: `e7460a8 Sync H100 rollout with persistent schedule pipeline`

## Current Status

The 5880 validation run showed the persistent-schedule bridge can keep the GPU
busy with multiple logical lanes on one physical GPU:

- result root: `out/massgen_full50_persistent_multiworker_5880/5548b12_w4_f16`
- report: `report_w4_compare/REPORT.md`
- videos: `run_w4`, `2200` MP4s
- chunks: `27/27` success
- capped frames: `35,199`
- wall time: `384.3s`
- average GPU utilization: `86.28%`
- peak VRAM: `15.60 GiB`
- full-run worker plot:
  `report_w4_compare/assets/graphs/full_run_stage_overlay.png`

The H100 wrapper now defaults to this persistent-schedule architecture instead
of the old family-level fanout:

```sh
python scripts/massgen/run_family_rollout_h100.py --pipeline-mode persistent ...
```

The old fanout remains available as:

```sh
python scripts/massgen/run_family_rollout_h100.py --pipeline-mode legacy ...
```

## Architecture In This Branch

Main execution chain:

```text
scripts/massgen/run_family_rollout_h100.py
  -> scripts/massgen/plan_persistent_h100_schedule.py
  -> scripts/massgen/summarize_persistent_schedule_lengths.py
  -> scripts/massgen/run_persistent_h100_schedule.py
  -> scripts/massgen/report_persistent_h100_schedule_run.py
```

Important outputs:

```text
<results-root>/
  h100_persistent_run_config.json
  persistent_schedule.json
  aggregate_render_plan.json
  natural_length_projection.json
  assignment_cpu_cores.json
  run_persistent/
    benchmark_summary.json
    gpu_samples.jsonl
    render_records.jsonl
    worker_stage_markers.jsonl
    persistent_schedule_renders/
  report_persistent/
    REPORT.md
    assets/graphs/gpu_vram_timeline.png
    assets/graphs/full_run_stage_overlay.png
    assets/tables/full_run_stage_summary.csv
    assets/tables/full_run_worker_lanes.csv
  logs/
  mp4_files.txt
  mp4_count.txt
```

## Key Implementation Files

- `scripts/massgen/run_family_rollout_h100.py`
  - H100 entrypoint.
  - Defaults to persistent mode.
  - Builds logical worker lanes from `--gpu-devices`, `--jobs-per-gpu`, and
    optional `--max-workers`.
  - Writes `assignment_cpu_cores.json`.
  - Forwards CPU thread caps, optional `taskset`, ffmpeg overrides, clean/resume
    behavior, and report options.

- `scripts/massgen/run_persistent_h100_schedule.py`
  - Executes `h100_persistent_schedule.v1`.
  - Supports repeated physical GPU ids as logical assignment lanes such as
    `0_w00`, `0_w01`.
  - Writes preemptible `.tmp` outputs and promotes only successful chunks.
  - Samples GPU/VRAM and emits worker lifecycle markers.
  - Accepts `--assignment-cpu-cores-json` and `--cpu-threads-per-worker`.

- `scripts/massgen/report_persistent_h100_schedule_run.py`
  - Generates GPU/VRAM timeline, first-window stage overlay, and full-run worker
    lifecycle overlay.
  - Writes stage and worker-lane CSVs for report/table use.

- `docs/h100_massgen_pipeline.md`
  - Operator-facing H100 persistent runbook.

- `docs/h100_linux_mirror_runbook.md`
  - Linux mirror setup and smoke command.

- `containers/h100/Dockerfile`
  - Renderer-side CUDA/H100 container. This is separate from the Pathplanner
    CPU MassGen container.

- `containers/h100/entrypoint.sh`
  - Container preflight and H100 run entrypoint.

- `scripts/massgen/setup_h100_linux_mirror.sh`
  - Clone/update helper for the Linux mirror and optional package copy.

- `scripts/massgen/build_h100_container.sh`
  - Quick iterative build of `navdp-datagen-h100:massgen` from the live
    checkout.

- `scripts/massgen/build_h100_clean_mirror_image.sh`
  - Release-style image build for Linux/H100. Exports a tracked-only
    superproject Docker context, avoids renderer submodule contents, installs
    `gsplat` through pip, checks clean git state, targets `linux/amd64`, and
    can optionally save the image tarball.

- `scripts/massgen/run_h100_container.sh`
  - Runs the H100 persistent pipeline inside the container with GPU, IPC, and
    common data root mounts.

- `docs/h100_persistent_gpu_worker_handoff.md`
  - Architecture notes for the later true persistent per-GPU process / CUDA IPC
    phases.

## Smoke Command For Linux H100 Mirror

Use a capped run before full natural-length rendering:

```sh
cd /mnt/<h100-data>/dongjk/navdp_data/Navdp_Datagen

/path/to/cuda-env/bin/python scripts/massgen/run_family_rollout_h100.py \
  --package-root /mnt/<h100-data>/dongjk/navdp_data/massgen_packages/<package> \
  --results-root /mnt/<h100-data>/dongjk/navdp_data/h100_results/<run-id> \
  --python-bin /path/to/cuda-env/bin/python \
  --gpu-devices 0,1,2,3 \
  --cpu-cores 120 \
  --jobs-per-gpu 4 \
  --video-backend cpu \
  --renders-per-family-source-scene 50 \
  --minimal-frames 16 \
  --command-attempts 3 \
  --clean
```

For resume after interruption, rerun the same command without `--clean`.
`--resume` is enabled by default.

## Container Smoke Command

Build on the Linux H100 mirror:

```sh
cd /mnt/<h100-data>/dongjk/navdp_data/Navdp_Datagen
IMAGE_TAG=navdp-datagen-h100:massgen scripts/massgen/build_h100_clean_mirror_image.sh
```

Run:

```sh
PACKAGE_ROOT=/mnt/<h100-data>/dongjk/navdp_data/massgen_packages/<package> \
RESULTS_ROOT=/mnt/<h100-data>/dongjk/navdp_data/h100_results/<run-id> \
GPU_DEVICES=0,1,2,3 \
CPU_CORES=120 \
JOBS_PER_GPU=4 \
MINIMAL_FRAMES=16 \
RENDERS_PER_FAMILY_SOURCE_SCENE=50 \
EXTRA_H100_ARGS="--clean" \
scripts/massgen/run_h100_container.sh run
```

If package paths reference nonstandard absolute roots, mount those roots into
the container before running.

## Local Validation Already Run

```sh
/Users/dongjk/miniconda3/bin/python3.13 -m py_compile \
  scripts/massgen/run_family_rollout_h100.py \
  scripts/massgen/run_persistent_h100_schedule.py

/Users/dongjk/miniconda3/bin/python3.13 -m pytest \
  tests/test_h100_family_rollout.py \
  tests/test_persistent_h100_schedule_runner.py \
  tests/test_persistent_scheduler.py
```

Result: `27 passed`.

Planning-only H100 wrapper smoke was run locally against an old package. It
correctly produced config/schedule artifacts, but no executable chunks because
the package referenced Linux scene roots such as `/mnt/DATA/...` that do not
exist on the Mac. This is expected; executable chunk generation must be checked
on the Linux mirror where scene and avatar roots exist.

## Current Caveats

- This is still a Phase-A bridge. Each chunk launches the existing renderer
  process; it is not yet the true single persistent renderer process per GPU.
- Multi-robot overlay plans remain blocked in the Phase-A schedule runner.
- H100 has no NVENC; use `--video-backend cpu`.
- The package must reference paths that exist on the target host:
  scene roots, `occupancy.json`, Gaussian PLYs, human avatar PLY sequences, and
  robot/AMO assets.
- The first production H100 run should keep `--jobs-per-gpu 4`, then tune up
  only if the full-run report shows GPU average below 80% and CPU/RAM/VRAM
  headroom remains.

## Next Session Checklist

1. Pull latest `massgen`.
2. Set up the Linux mirror under the target H100 data mount.
3. Verify `nvidia-smi`, CUDA/PyTorch, TeleSim import, scene roots, and
   `human_gs_source`.
4. Build or enter the H100 renderer container if using containerized execution.
5. Run the capped H100 persistent smoke.
6. Download `report_persistent/`, `run_persistent/benchmark_summary.json`,
   `natural_length_projection.json`, and a representative sample of videos.
7. Decide whether to increase `--jobs-per-gpu` or move on to full natural
   length.
