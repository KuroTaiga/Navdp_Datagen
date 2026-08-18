# H100 MassGen Rollout Pipeline

Use `scripts/massgen/run_family_rollout_h100.py` for the compute platform with H100 GPUs, 120 CPU cores, and large system memory. Keep using the existing 5880 rollout path for 5880.

The H100 profile is separate because H100 does not provide NVENC/RT video encode blocks. The pipeline therefore keeps Gaussian rendering on CUDA, forces video output through CPU `libx264`, and raises GPU utilization by running multiple isolated render processes per GPU.

## Default Strategy

- Video backend: `cpu`
- Actor placement: `--actor-gpu-resident` enabled by default
- GPU fanout: 4 render processes per GPU by default
- CPU budget: `floor(cpu_cores / worker_count)` cores per render process
- CPU control: `taskset` on Linux when available, plus `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `TORCH_NUM_THREADS`
- Monitoring: `nvidia-smi` samples are written to `gpu_utilization.jsonl` and summarized in `gpu_utilization_summary.json`
- Soft target: average sampled GPU utilization at or above 80%

## Example

```sh
python scripts/massgen/run_family_rollout_h100.py \
  --package-root /path/to/massgen_package \
  --results-root /path/to/h100_results \
  --python-bin /path/to/env/bin/python \
  --gpu-devices 0,1,2,3 \
  --cpu-cores 120 \
  --jobs-per-gpu 4 \
  --video-backend cpu \
  --target-gpu-util 80 \
  --limit 1 \
  --retry 1
```

Each family still gets its own copied package folder, render outputs, logs, metrics, `time.txt`, and summary rows under:

```text
<results-root>/
  family_render_summary.json
  family_render_summary.jsonl
  gpu_utilization.jsonl
  gpu_utilization_summary.json
  logs/
  families/<family>/
```

## Tuning

Start with `--jobs-per-gpu 4`. If `gpu_utilization_summary.json` reports average utilization below 80% and CPU/RAM headroom remains, increase to 6 or 8 jobs per GPU. If CPU encode time dominates or the host starts swapping, reduce `--jobs-per-gpu` or set `--cpu-cores-per-worker` explicitly.

For quick before/after comparisons, use `--minimal-frames` on the same package and compare `duration_total_sec`, `time_per_frame_sec`, `gpu_utilization_summary.json`, and per-family render logs. Full-quality runs should omit `--minimal-frames`.

## Robot Overlay Scheduling

Multi-robot missions are split as one render job per viewpoint robot. Each
viewpoint job renders the Gaussian scene once, then chains peer robot GLB
overlays into a final `__with_peer_robots.mp4` track. The G1 walking posture is
driven by Kimodo SMPL-X frames retargeted to G1 joint poses.

This overlay stage is not the main GPU-saturation lever. In the current
implementation, Gaussian rendering is the CUDA-heavy phase; GLB robot overlays
are short pyrender/EGL subprocesses plus image writes. For H100 runs, keep the
80% GPU-utilization target focused on concurrent GS render workers:

- run several independent render jobs per GPU;
- force base video encoding to CPU (`--video-backend cpu`);
- schedule robot overlays after base frames exist, or in a separate post-render
  pool if overlay latency starts blocking new GS jobs;
- prefer direct env Python (`/path/to/env/bin/python`) over wrapping the executor
  in `conda run`, because repeated EGL subprocesses were more stable without the
  wrapper during the 5880 smoke.

The immediate optimization target is therefore worker fanout and CPU encode
allocation, not making each individual robot-overlay command heavier.
