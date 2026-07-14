# NavDP Path Renderer Release

Lean render-only entry point for turning planned paths and one Gaussian splatting scene into MP4s, optional RGB frames, depth maps, and camera metadata.

## Contents

This folder is the release package. It includes:

```text
README.md
install.sh
render_path.sh
render_path.py
unpack_3dgs_compressed.py
render_label_paths_telesim.py
gaussian_renderer/
arguments/
scene/
utils/
lighting/
TeleSim3D/tele_sim/
```

Only the render-time TeleSim subset is bundled; planning, Habitat, LLM, and viewer tools are intentionally omitted.

## Install

Recommended conda install:

```bash
conda create -n navdp-render python=3.10
conda activate navdp-render
./install.sh
```

When run inside an active non-base conda env, `install.sh` installs into that conda env.

Venv install, for machines not using conda:

```bash
USE_VENV=true ./install.sh
source .venv-navdp-render/bin/activate
```

Defaults:

- Current conda env when `CONDA_PREFIX` is active; otherwise Python venv at `.venv-navdp-render`
- PyTorch `2.5.1` CUDA `cu121`
- `gsplat` from pip
- editable install of the bundled `TeleSim3D/tele_sim`

Overrides:

```bash
ENV_DIR=/path/to/venv TORCH_CUDA=cu124 ./install.sh
INSTALL_TORCH=false ./install.sh
USE_VENV=true ./install.sh
```

GPU rendering requires a CUDA-capable PyTorch install. Video output uses `--video-backend nvenc` by default and falls back through ImageIO/ffmpeg behavior; use `--video-backend cpu` when NVENC is unavailable.

The render path uses `gsplat` by default. The legacy `diff_gaussian_rasterization` and `simple_knn` extensions are not required for render-only gsplat usage.

## Expected Inputs

Scene directory:

```text
scene_id/
  occupancy.json
  occupancy.png
  3dgs_raw.ply          # preferred
```

If the PLY has another name, pass `--gaussian-model /path/to/model.ply`.

## Unpack Compressed Scenes

Some scene packages use a truly packed PLY schema with fields such as
`packed_position`, `packed_rotation`, `packed_scale`, and `packed_color`. The
renderer can load those directly, but you can materialize a standard GraphDeco
PLY first:

```bash
python unpack_3dgs_compressed.py /data/scenes/0001_839920
```

This writes:

```text
/data/scenes/0001_839920/3dgs_decompressed.ply
```

For a parent directory of many scene folders:

```bash
python unpack_3dgs_compressed.py /data/CHINGMU_rescaled_1 --recursive
```

For an explicit PLY path:

```bash
python unpack_3dgs_compressed.py \
  /data/CHINGMU_rescaled_1/0024_858856/3dgs_compressed.ply \
  --output /data/CHINGMU_rescaled_1/0024_858856/3dgs_decompressed.ply
```

The unpacker uses this package's `GaussianModel.load_ply()` and currently
requires CUDA because that loader allocates tensors on CUDA. If a
`3dgs_compressed.ply` is already an expanded float-field PLY, the script reports
`[SKIP] already standard PLY`; use it as-is or pass `--copy-standard` if you
want to rewrite it anyway.

Path JSON:

```json
{
  "path": {
    "raster_world": [{"x": 0.0, "y": 0.0, "z": 0.0}],
    "raster_pixel": [[123, 456]]
  }
}
```

`raster_world` and `raster_pixel` must have the same length.

## Render One Path

```bash
./render_path.sh \
  --scene-dir /data/scenes/0001_839920 \
  --path-json /data/tasks/0001_839920/label_paths/100.json \
  --output-dir /tmp/navdp_render
```

Outputs:

```text
/tmp/navdp_render/<scene_id>/<label_id>.mp4
/tmp/navdp_render/<scene_id>/<label_id>_camera.json
```

Useful flags:

```bash
--label-id 100
--scene-id 0001_839920
--resolution 1280 720
--video-backend cpu
--rgb-frames
--save-depth-maps
--minimal-frames 20
```

Renderer-specific flags can be appended directly; unknown wrapper flags are passed through to `render_label_paths_telesim.py`.

## Render All Paths

```bash
./render_path.sh \
  --scene-dir /data/scenes/0001_839920 \
  --paths-dir /data/tasks/0001_839920 \
  --output-dir /tmp/navdp_render
```

The wrapper scans `--paths-dir` for renderable path JSONs, skips helper JSONs, and excludes `*_detailed.json` by default. Use `--no-exclude-detailed-labels` only if detailed path files should be rendered too.

`--video-fps` defaults to `10`, matching `render_label_paths_telesim.py` and the TeleSim batch scripts.

## Coordinate Conversion

The renderer maps planned-path coordinates into the Gaussian splat world per path:

1. Read `path.raster_world` and `path.raster_pixel`.
2. Convert pixels to scene-map world coordinates using `occupancy.json` (`min`, `max`, `scale`) and `occupancy.png` size.
3. Fit an axis-aligned affine transform from `raster_world` to that scene-map world.
4. Apply optional coordinate fixes.
5. Mirror translation around the occupancy-map center by default.

Special cases:

- `--path-handedness left` is the default.
- `--path-handedness right` flips Y before the affine fit.
- `--swap-xy` handles path producers that wrote X/Y in the opposite order.
- `--negate-xy` handles a full XY sign mismatch.
- `--no-mirror-translation` should be used when the path JSON is already in the same world convention as the Gaussian scene.
- `--follow-distance 0.0` is the FPV default. Increase it only for follow-camera renders.

The default backend is `gsplat` through `GAUSSIAN_RENDER_BACKEND=gsplat`. Use `--backend diff-gaussian` only when the legacy rasterizer is installed and needed.
