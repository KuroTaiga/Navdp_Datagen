# Lighting Pipeline

This folder contains scripts and utilities for lighting adjustments on Gaussian-splatting scenes.

## Darken/Brighten Scene PLYs

Create a new scenes folder with scaled SH coefficients:

```bash
python lighting/darken_scenes.py \
  --scenes-dir data/scenes \
  --output-dir data/scenes_dark \
  --ev -1.5 \
  --other-mode link \
  --overwrite \
  --report-json lighting/darken_report.json
```

- `--ev` is exposure delta (`scale = 2**ev`). Negative values darken.
- `--other-mode link` symlinks non-PLY assets to save space.

## Render-Time Camera Light (CL)

Enable depth-normal shading and optional shadow mapping:

```bash
python render_label_paths.py \
  --scenes-dir data/scenes_dark \
  --tasks-dir data/interiorGS_0500_42 \
  --scene 0004_840011 --label-id 100 \
  --cl-enable \
  --cl-strength 1.0 \
  --cl-color 1 1 1 \
  --cl-ambient 0.2 \
  --cl-diffuse 1.0 \
  --cl-specular 0.2 \
  --cl-shininess 16 \
  --cl-range 8 \
  --cl-offset 0 0 0.2 \
  --cl-shadow \
  --cl-shadow-bias 0.02 \
  --cl-shadow-strength 0.2
```

Notes:
- `--cl-shadow` triggers a second render pass from the light POV.
- If the light offset is zero, shadows will be minimal (light == camera).

## Evaluation Helper (10 Paths)

Baseline vs post-filter vs render-time lighting:

```bash
bash lighting/lighting_eval_10paths.sh
```

This writes reports under `analysis/lighting_eval`.
