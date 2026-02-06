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

## Lower Lighting MP4s (Post vs Render-Time)

Post-process an existing MP4 (fast, no re-render):

```bash
python lighting/apply_light_filter_mp4.py \
  data1/0500_fpv_npc \
  --pattern "*.mp4" \
  --output-dir analysis/lower_light_mp4 \
  --light-mode global \
  --light-strength -0.35 \
  --output-json analysis/lower_light_mp4/report.json
```

Render-time darkening (during rendering):

```bash
python render_label_paths.py \
  --scenes-dir data/scenes \
  --tasks-dir data/interiorGS_0500_42 \
  --scene 0004_840011 --label-id 100 \
  --light-mode global \
  --light-strength -0.35
```

For a consistent, reportable lighting delta, compare `luma_log2_mean` between reports
(delta in log2 mean ≈ EV change).

## Build Lighting Datasets from Existing MP4s

Create new dataset folders with globally scaled brightness while keeping filenames and structure:

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --output-json analysis/lighting_dataset_report.json
```

Notes:
- Default `--scan-mode stream` starts processing while MP4s are still being discovered.
- Use `--scan-mode sorted` to match the old “scan everything first” behavior (required for `--suffix-mode luma` / base-luma computation).

Parallel processing with progress:

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --workers 8 \
  --progress-every 25
```

Progress JSON updates:

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --workers 8 \
  --progress-every 25 \
  --progress-json analysis/lighting_dataset_progress.json
```

Live log file:

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --workers 8 \
  --progress-every 25 \
  --log-file analysis/lighting_dataset.log
```

Faster MP4 discovery (from a list file):

```bash
find ./data2/0500_fpv -name "*.mp4" > analysis/mp4_list.txt
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --mp4-list analysis/mp4_list.txt \
  --workers 8 \
  --progress-every 25
```

Scan progress (while enumerating files):

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --scan-progress-every 5000
```

Skip base-luma computation (process immediately):

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --suffix-mode ev \
  --workers 8 \
  --progress-every 25
```

Provide a precomputed base luma (for `--suffix-mode luma`):

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --suffix-mode luma \
  --base-luma 0.46
```

## Time-of-Day Lighting Tones

Generate multiple datasets with different color temperatures and brightness levels to simulate
daylight hours:

```bash
python lighting/build_time_of_day_dataset.py ./data2/0500_fpv \
  --presets dawn noon dusk \
  --output-root ./data2 \
  --workers 8
```

Notes:
- Default `--scan-mode stream` starts processing immediately.
- `--max-files N` now stops discovery early (useful for quick tests).

Outputs are written to folders named like `./data2/0500_fpv_dawn`. Use `--preset-json` to supply
custom tone settings.
Built-in presets: `dawn`, `morning`, `noon`, `afternoon`, `golden_hour`, `dusk`, `blue_hour`, `night`.

## CHINGMU_0800 HDD Helpers

For HDD-backed navdata (symlinked at `./navdata`), use the helper scripts below. They reuse a
cached MP4 list and default to low worker counts for HDD-friendly IO.

```bash
bash lighting/run_chingmu_0800_hdd_time_of_day.sh
bash lighting/run_chingmu_0800_hdd_luma.sh
```

Override paths and settings via env vars such as `INPUT_DIR`, `OUTPUT_ROOT`, `WORKERS`,
`SUFFIX_MODE`, and `BASE_LUMA`.

## Time-of-Day Method Comparison (Render vs MP4)

Compare golden/blue hour lighting between render-time filtering and MP4 post-processing:

```bash
bash lighting/run_time_of_day_compare.sh
```

Override tone strengths and temperatures via env vars (`GOLDEN_STRENGTH`, `GOLDEN_TEMP_K`,
`BLUE_STRENGTH`, `BLUE_TEMP_K`) or set `TONES="golden_hour blue_hour"`.

Defaults:
- Skips base luma unless `--suffix-mode luma`, `--compute-base-luma`, or `--base-only` is set.
- Generates three outputs: 1.5x brighter, 0.5x darker, 0.2x darker.
- Output folders are suffixed by scale (e.g., `./data2/0500_fpv_1.5L`).
- Non-MP4 files are skipped by default; use `--other-mode copy` to copy after MP4 processing.

Alternate suffix modes:
- `--suffix-mode luma` -> `_0.300`
- `--suffix-mode ev` -> `_EVm1.00` / `_EVp0.58`

Base-luma only (sample 10 MP4s per scene):

```bash
python lighting/build_lighting_dataset.py ./data2/0500_fpv \
  --base-sample-per-scene 10 \
  --base-max-scenes 50 \
  --base-only
```

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
  --cl-shadow-strength 0.2 \
  --cl-normal-smooth 2 \
  --cl-shadow-pcf 1
```

Notes:
- `--cl-shadow` triggers a second render pass from the light POV.
- If the light offset is zero, shadows will be minimal (light == camera).
- `--cl-offset` uses camera coordinates: X right, Y up, Z forward.
- To keep the light at camera height, use `--cl-offset 0 0 0` and control height with `--height-offset` (0.0 -> ~1.0m, 0.3 -> ~1.3m).

## Evaluation Helper (10 Paths)

Baseline vs post-filter vs render-time lighting:

```bash
bash lighting/lighting_eval_10paths.sh
```

This writes reports under `analysis/lighting_eval`.

## Driver Scripts

Run the variants on the same 10 sampled paths:

```bash
bash lighting/run_lighting_variants.sh
```

Generate luma + speed summaries and comparison outputs:

```bash
bash lighting/run_lighting_reports.sh
```
