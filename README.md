# NavDP Gaussian Splatting Datagen

This repository generates navigation datasets (FPV and follow-camera) from 3D Gaussian Splatting scenes. It supports animated human actors, NPC placement, per-frame camera metadata, and MP4 video output with GPU or CPU encoding backends.

## Repository Layout
- `render_label_paths.py`: core renderer (paths → frames/video/metadata).
- `parallel_render_paths.py`: sharded runner for parallel jobs.
- `run_random_fpv_datagen.sh`: FPV dataset pipeline.
- `run_random_human_datagen.sh`: follow-camera pipeline with actors.
- `scripts/quick_pipeline_test.py`: small end-to-end test + resource report.
- `scripts/render_glb_robot_overlay.py`: optional GLB robot foreground compositor for existing rendered frames.
- `navdp_datagen_pipeline.md`: pipeline overview and detailed usage.
- `docs/scene_placement_orientation.md`: coordinate, floor-plane, yaw, and camera-basis reference for placing foreground objects.
- `docs/`: design notes and scheduling plans.

## Setup
1) Create and activate the conda environment:
```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```
2) Ensure CUDA is available (GPU required for rendering).

## Inputs
- Scenes: `data/scenes/<scene_id>/` (PLY + occupancy metadata).
- Paths: `data/interiorGS_0500_42/<scene_id>/` (label JSONs).
- Actor PLYs: `data/SHHQ_gs/walking/` or `data/human_gs_source/`.

## Common Workflows
FPV dataset:
```bash
./run_random_fpv_datagen.sh
```

Follow-camera dataset with actors:
```bash
./run_random_human_datagen.sh
```

Single-scene render:
```bash
python render_label_paths.py \
  --scenes-dir data/scenes \
  --tasks-dir data/interiorGS_0500_42 \
  --scene 0001_839920 \
  --label-id 100 \
  --output-dir data1/debug_render \
  --video --save-camera-metadata
```

Overlay a GLB robot on an existing GS render:
```bash
python scripts/convert_urdf_visuals_to_glb.py \
  --urdf data/g1_description/g1_29dof_mode_16.urdf \
  --output assets/robots/g1_29dof_mode_16.glb

python scripts/render_glb_robot_overlay.py \
  --camera-json data1/debug_render/0001_839920/100_camera.json \
  --frames-dir data1/debug_render/0001_839920/100 \
  --robot-glb assets/robots/g1_29dof_mode_16.glb \
  --robot-urdf data/g1_description/g1_29dof_mode_16.urdf \
  --poses-json data1/debug_render/0001_839920/100_robot_poses.json \
  --output-dir data1/debug_render/0001_839920/100_robot \
  --compose-mode foreground
```

Use `--compose-mode depth` when saved depth maps should occlude the GLB robot behind GS geometry. The pose JSON can be produced by IMO/AMO or another robot controller and should provide per-frame `position` plus `yaw_deg`/`yaw_rad`, or a full 4x4 `transform`. For articulated AMO control, include `joint_positions`/`joints` dictionaries or list-valued `amo_pose`/`qpos` with `joint_names`; `--robot-urdf` enables those joints to drive the GLB mesh links.

Generate a 10-path G1 robot follow-camera example from the first `0001_*` scene:
```bash
python scripts/run_g1_robot_follow_example.py \
  --tasks-dir data/interiorGS_0500_42 \
  --output-dir data2/g1_robot_follow_example \
  --path-count 10
```

Quick test (few labels + resource sampling):
```bash
python scripts/quick_pipeline_test.py --scene 0001_839920 --label-count 3
```

## Backend Defaults and Flags
Rendering defaults use GPU transforms and NVENC video if available, with CPU fallback:
- PLY transforms: `--ply-transform-backend gpu|cpu`
- Video backend: `--video-backend nvenc|cpu`
- NVENC tuning: `--video-nvenc-preset`, `--video-nvenc-bitrate`

CPU encoding is explicitly H.264 (`libx264`). GPU encoding is H.264 (`h264_nvenc`). If a backend is unavailable, the pipeline logs a warning and falls back to CPU.

## NVENC-enabled FFmpeg (optional)
Conda FFmpeg builds typically do not include NVENC. If you want GPU video encoding, install an NVENC-capable system FFmpeg and point ImageIO to it.

```bash
# Build deps (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential pkg-config yasm nasm \
  libx264-dev libx265-dev libnuma-dev git

# NVENC headers
git clone https://github.com/FFmpeg/nv-codec-headers.git
cd nv-codec-headers
make
sudo make install
cd ..

# FFmpeg with NVENC
git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
./configure \
  --enable-gpl --enable-nonfree \
  --enable-cuda-nvcc --enable-nvenc \
  --enable-libx264 --enable-libx265 \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-ldflags=-L/usr/local/cuda/lib64
make -j"$(nproc)"
sudo make install

# Verify encoder availability
ffmpeg -hide_banner -encoders | grep nvenc

# Point ImageIO to system FFmpeg if needed
export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# Persist across new shells (optional)
echo 'export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg' >> ~/.bashrc
source ~/.bashrc
```

## Acknowledgments
This codebase builds on the original 3D Gaussian Splatting project by Kerbl et al. (GraphDeco/Inria). We thank the original authors and contributors for their foundational work.
