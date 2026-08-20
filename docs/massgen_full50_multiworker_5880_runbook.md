# MassGen Full50 Multiworker 5880 Runbook

This runbook prepares the next capped full50 render test on 5880 with multiple
logical render workers on the same physical GPU.

Do not start the render until explicitly requested.

## What Changed

- `plan_persistent_h100_schedule.py` supports `--workers-per-gpu`.
- The persistent scheduler can emit several assignment lanes for one physical
  GPU, e.g. `0_w00`, `0_w01`, `0_w02`, `0_w03`, all with `gpu_id=0`.
- `run_persistent_h100_schedule.py` executes those lanes concurrently when
  `--workers` matches the lane count, but still sets renderer subprocess
  `CUDA_VISIBLE_DEVICES` to the physical GPU id.
- `report_persistent_h100_schedule_run.py` generates a report, GPU/VRAM plots,
  first-window lifecycle lane visualization, CSV tables, and JSON metrics.
- `summarize_persistent_schedule_lengths.py` records capped and natural path
  length stats for projection.

## Remote Defaults

The launcher is:

```sh
scripts/massgen/run_full50_persistent_multiworker_5880.sh
```

Default settings:

```text
WORKERS=4
FRAMES=16
GPU_ID=0
SAMPLE_INTERVAL_SEC=0.1
COMMAND_ATTEMPTS=3
GROUP_MAX_LABELS_PER_COMMAND=0
MAX_ITEMS_PER_CHUNK=0
PYTHON_BIN=/home/dongjk/.conda/envs/cuda121/bin/python
PACKAGE_ROOT=/mnt/DATA1/dongjk/navdp_data/h100_phase1_baseline_9891557/package_baseline_50
REMOTE_DATA_ROOT=/mnt/DATA1/dongjk/navdp_data
```

The default output root is:

```text
/mnt/DATA1/dongjk/navdp_data/massgen_full50_persistent_multiworker_<git>_w<workers>_f<frames>
```

## Run Command

From the 5880 repo root:

```sh
WORKERS=4 FRAMES=16 CLEAN=1 \
  bash scripts/massgen/run_full50_persistent_multiworker_5880.sh
```

For a planning-only dry preparation without rendering:

```sh
WORKERS=4 RUN_RENDER=0 RUN_ANALYSIS=0 \
  bash scripts/massgen/run_full50_persistent_multiworker_5880.sh
```

## Expected Outputs

Remote:

```text
<ROOT>/
  run_config.env
  persistent_schedule.json
  aggregate_render_plan.json
  natural_length_projection.json
  logs/
  run_w<WORKERS>/
    benchmark_summary.json
    gpu_samples.jsonl
    render_records.jsonl
    worker_stage_markers.jsonl
    persistent_schedule_renders/
  report_w<WORKERS>/
    REPORT.md
    assets/graphs/gpu_vram_timeline.png
    assets/graphs/first_window_stage_overlay.png
    assets/tables/*.csv
    metrics/summary_metrics.json
```

## Download After Run

After the render completes, archive only the useful outputs:

```sh
ROOT=/mnt/DATA1/dongjk/navdp_data/massgen_full50_persistent_multiworker_<git>_w4_f16
ssh codex-5880 "cd \"$ROOT\" && tar -czf /tmp/massgen_full50_multiworker_w4_f16.tgz \
  run_config.env df_before.txt df_after.txt nvidia_smi_before.txt nvidia_smi_after.txt \
  persistent_schedule.json aggregate_render_plan.json natural_length_projection.json \
  logs run_w4 report_w4"
mkdir -p out/massgen_full50_persistent_multiworker_5880/w4_f16
scp codex-5880:/tmp/massgen_full50_multiworker_w4_f16.tgz \
  out/massgen_full50_persistent_multiworker_5880/w4_f16/
tar -xzf out/massgen_full50_persistent_multiworker_5880/w4_f16/massgen_full50_multiworker_w4_f16.tgz \
  -C out/massgen_full50_persistent_multiworker_5880/w4_f16/
ssh codex-5880 "rm -f /tmp/massgen_full50_multiworker_w4_f16.tgz"
```

If the remote report fails due to Matplotlib availability, run local analysis
after download:

```sh
python3 scripts/massgen/report_persistent_h100_schedule_run.py \
  --run-root out/massgen_full50_persistent_multiworker_5880/w4_f16/run_w4 \
  --output-root out/massgen_full50_persistent_multiworker_5880/w4_f16/report_w4 \
  --title "MassGen Full50 Persistent Multiworker 5880 w4" \
  --stage-window-min 4 \
  --natural-length-json out/massgen_full50_persistent_multiworker_5880/w4_f16/natural_length_projection.json
```

## Comparison Target

Compare against the single-worker run:

```text
out/massgen_full50_persistent_5880/e7c239b_f16/FULL50_PERSISTENT_5880_REPORT.md
```

Primary deltas:

- wall time;
- average GPU utilization;
- samples at or above 80%;
- first 4-minute actor-load downtime;
- per-lane render-loop overlap;
- setup seconds per frame;
- output size and failure/retry count.
