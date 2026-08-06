# Datagen Demo Video Source Plan

This file lists the source clips needed to produce two storytelling videos:
1) `Human Asset Datagen Pipeline`
2) `Scene Creation + Goal + Trajectory + Moving Human`

## 1) Human Asset Datagen Pipeline

### Source Clip Checklist
| Shot ID | What to show | Source clip/data needed | Tool/source owner | Target length |
|---|---|---|---|---|
| H1 | Input human image | Original portrait/full-body image | Dataset/raw capture | 1-2s |
| H2 | SMPLify extraction | Screen capture or render of fitted SMPL-X mesh over image | External SMPLify stage | 2-3s |
| H3 | LHM canonical avatar | Turntable or multi-view render of LHM canopose 3DGS avatar | External LHM stage | 2-3s |
| H4 | SMPL-X to LHM fit optimization | Before/after overlay (SMPL-X misaligned vs optimized fit) | Your fitting/optimization script | 2-3s |
| H5 | Motion extraction | Input animation source (waving/running/walking) + extracted motion curves/skeleton | Motion extraction stage | 2-3s |
| H6 | Motion-driven SMPL-X | Animated SMPL-X replay using extracted motion | Motion retargeting stage | 2-3s |
| H7 | Motion-driven 3DGS human | Final animated 3DGS human in scene/follow camera | `render_label_paths.py` outputs | 3-4s |

### Must-have output artifacts before editing
- `input_image.(png/jpg)`
- `smplify_fit_preview.mp4` (or image sequence)
- `lhm_canopose_preview.mp4`
- `smplx_to_lhm_fit_before_after.mp4`
- `motion_source_clip.mp4` and `motion_extracted_preview.mp4`
- `smplx_driven_animation.mp4`
- `3dgs_driven_animation_in_scene.mp4`

## 2) Scene Creation + Goal + Trajectory + Moving Human

### Source Clip Checklist
| Shot ID | What to show | Source clip/data needed | Tool/source owner | Target length |
|---|---|---|---|---|
| S1 | Empty scene | Static scene render (no walls/chairs/items) | Scene editor / renderer | 2s |
| S2 | Progressive scene placement | Time-lapse: walls, chairs, and objects being added | Scene creation pipeline | 3-4s |
| S3 | Goal highlight | Item highlight (outline/glow/marker) for target object | Scene editor + compositor | 2s |
| S4 | Planned robot trajectory | Top-down/BEV trajectory overlay from start to goal | Path planner + BEV debug output | 2-3s |
| S5 | Robot camera run | FPV/follow render of robot moving along planned path | `render_label_paths.py` output MP4 | 3-4s |
| S6 | Insert moving human | Same scene with animated human actor added and moving | `render_label_paths.py` with actor sequence | 3-4s |
| S7 | Final combined shot | Goal + trajectory + moving human together (short hero shot) | Editor composite | 2-3s |

### Useful repo-side sources for S4-S6
- Path JSONs: `data/interiorGS_0500_42/<scene>/<label>.json`
- Scene assets: `data/scenes/<scene>/`
- Actor sequences: `data/SHHQ_gs/walking/` (or your actor root)
- Render entrypoints:
  - `python render_label_paths.py ... --video --show-BEV`
  - `python scripts/render/views/render_first_frame.py --overwrite --verbose` (for quick static previews)

## Editing Order (recommended)
1. Build each video from short stage clips (H1->H7 and S1->S7).
2. Keep each stage as one visual claim (2-4s max).
3. Add short on-screen labels only (tool names + one action verb).
4. End each video with final integrated result shot.

## Optional: fast dataset mosaic opener
Use `scripts/media/make_random_video_mosaic.py` to produce a short 16x10 random-grid opener from `./data2/0500_fpv` before the pipeline narrative starts.
