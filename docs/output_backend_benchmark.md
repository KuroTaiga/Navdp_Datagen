# Output Backend Benchmark

Use `scripts/render/benchmark_output_backends.py` to time TeleSim rendering,
photo/depth writes, and video backends with the same scene/path input.

The helper writes every run into a separate folder, then produces:

- `benchmark_summary.json`: machine-readable commands, timings, logs, metrics,
  and output video paths.
- `benchmark_report.md`: tables for run times, per-step totals, backend deltas,
  and MP4 outputs.
- `runs/<mode>__<backend>/.../*.mp4`: one output video per video backend/mode.
- `logs/<mode>__<backend>.log`: renderer stdout/stderr for failures.

## Timed Steps

TeleSim path metrics now include:

- `actor_visibility_sec`: actor frustum culling.
- `actor_transform_sec`: human PLY transform into the current world pose.
- `actor_tensor_pack_sec`: actor data conversion to renderer tensors.
- `actor_merge_update_sec`: scene+actor Gaussian model create/update.
- `gaussian_render_sec`: Gaussian render call.
- `gpu_readback_sec`: GPU tensor to CPU RGB readback for saved PNGs.
- `perframe_light_sec`: light filter and camera-light shading.
- `perframe_depth_sec`: depth map writes.
- `perframe_png_sec`: RGB PNG writes.
- `mp4_write_sec`: imageio CPU/NVENC frame append time.
- `h264_encode_sec`: GPU video encode time.
- `h264_mux_sec`: GPU H.264 bitstream mux to MP4.
- `video_close_sec`: imageio CPU/NVENC writer close/flush time; GPU close is
  split into `h264_encode_sec` and `h264_mux_sec`.

The old `render` and `encode` aliases remain in `stage_seconds` for existing
smoke scripts.

## Remote Run On 5880 Host

Local config has a `5880host` entry in ignored/private config files. Do not copy
those values into tracked docs or scripts.

On the remote host:

```bash
ssh 5880host
cd /home/dongjk/project_files/NavDp_Jiankun_ver/navdp_api/gaussian_splatting
```

Before switching branches, inspect and preserve any remote-local work:

```bash
git status --short --branch --untracked-files=all
git diff --stat
git ls-files --others --exclude-standard
```

If the dirty files are source/config changes that should be preserved, commit
only those reviewed files:

```bash
git add <reviewed-source-file> <reviewed-config-file>
git commit -m "Save local render host state before output benchmark"
```

Do not blindly `git add -A` if generated outputs, data dumps, credentials, or
machine-local env files are present. Move generated outputs under an ignored
output directory or leave them in place if they do not block branch switching.

Then update the benchmark branch:

```bash
git fetch origin
git switch massgen || git switch -c massgen origin/massgen
git pull --ff-only
```

Run the benchmark. Adjust `--scenes-dir`, `--tasks-dir`, and `--scene` for the
remote dataset layout. If `--scene` is omitted, the helper uses the first scene
name shared by both input directories.

```bash
python3 scripts/render/benchmark_output_backends.py \
  --scenes-dir ./data/CHINGMU_scenes_rescaled \
  --tasks-dir ./data/CHINGMU_75_rescaled_0800_42_iter1 \
  --output-root /tmp/navdp_output_backend_benchmark \
  --max-labels 1 \
  --minimal-frames 120 \
  --backends nvenc,cpu,gpu \
  --modes video_only,video_rgb,video_depth,rgb_only,depth_only
```

By default, `STRICT_GPU_BACKENDS=1` is set for child renderer processes so NVENC
does not silently fall back to CPU. The `gpu` backend requires `PyNvVideoCodec`;
if it is missing, that run fails and the report keeps the failure log.

## Reading Results

Open:

```bash
less /tmp/navdp_output_backend_benchmark/benchmark_report.md
```

Use the report as the optimization guide:

- If `mp4_write_sec`, `h264_encode_sec`, or `video_close_sec` dominates, improve
  video encoding/output first.
- If `perframe_png_sec` or `perframe_depth_sec` dominates, reduce image/depth
  writes, batch/compress differently, or write to faster storage.
- If actor stages dominate, optimize human transform/tensor packing/merge before
  considering C++/CUDA.
- If `gaussian_render_sec` dominates, backend render performance is the main
  bottleneck.
