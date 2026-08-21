# H100 Linux Mirror Runbook

This runbook is for creating a Linux/H100 mirror of the MassGen renderer after
the persistent H100 pipeline has been validated on 5880.

## Goal

Keep the H100 platform checkout isolated from 5880-specific test state while
using the same `massgen` branch implementation:

- persistent scene/resource-aware planning;
- multiple logical worker lanes per physical H100;
- CPU `libx264` video encoding;
- taskset CPU affinity and per-worker thread caps;
- preemptible chunk outputs and resume markers;
- 10 Hz-ish GPU/VRAM sampling and full-run lifecycle reports.

## Suggested Layout

```text
/mnt/<h100-data>/dongjk/navdp_data/
  Navdp_Datagen/                  # git mirror, branch massgen
  massgen_packages/<package>/     # copied planner/render package
  h100_results/<run-id>/          # run output root
  human_gs_source/                # human avatar source, if not already elsewhere
```

Use a platform-local data mount for outputs. Do not write large results into
the git checkout.

## Mirror Setup

```sh
cd /mnt/<h100-data>/dongjk/navdp_data
git clone <repo-url> Navdp_Datagen
cd Navdp_Datagen
git switch massgen
git pull --ff-only origin massgen
```

Verify the environment:

```sh
/path/to/cuda-env/bin/python -m py_compile \
  render_label_paths_telesim.py \
  scripts/massgen/run_family_rollout_h100.py \
  scripts/massgen/plan_persistent_h100_schedule.py \
  scripts/massgen/run_persistent_h100_schedule.py \
  scripts/massgen/report_persistent_h100_schedule_run.py

/path/to/cuda-env/bin/python - <<'PY'
import torch
print("cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY

nvidia-smi
```

Also verify the package references existing scene and avatar roots. The
renderer does not copy scene Gaussian PLYs or human avatar PLY sequences into
the package; it expects referenced paths to exist on the H100 host.

## Smoke Command

Start with a capped run before a full natural-length run:

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

Outputs to inspect first:

```text
<results-root>/run_persistent/benchmark_summary.json
<results-root>/report_persistent/REPORT.md
<results-root>/report_persistent/assets/graphs/full_run_stage_overlay.png
<results-root>/report_persistent/assets/tables/full_run_worker_lanes.csv
<results-root>/mp4_count.txt
```

## Resume And Preemption

The persistent runner writes chunks into temporary directories and promotes
them only after success. If the VM is killed or GPUs must be released, rerun the
same command without `--clean`; `--resume` is enabled by default.

Use `--clean` only when intentionally discarding previous partial output.

## Full Run

For full natural-length generation, omit `--minimal-frames` and remove smoke
caps such as `--renders-per-family-source-scene`. Keep `--jobs-per-gpu 4` for
the first full run, then tune upward only if:

- GPU average is below 80%;
- host CPU/RAM are not saturated;
- CPU encode/write stages are not building a backlog;
- peak VRAM has enough margin for the selected scene/avatar mix.
