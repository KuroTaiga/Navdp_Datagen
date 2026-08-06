# Script Cleanup and Refactor Plan

Date: 2026-08-05
Branch: `massgen`

This document tracks the remaining script cleanup work. The goal is to make the
repo easy to understand from the top level: current entry points should be few,
documented, and backed by reusable Python modules; niche/debug scripts should be
deleted, archived outside the repo, or moved under an explicitly named area.

## Scope

This cleanup targets first-party NavDP Datagen code and first-party release
packaging. Leave vendored or upstream code alone unless a separate dependency
update calls for it:

- `submodules/`
- `SIBR_viewers/`
- `lpipsPyTorch/`
- original GraphDeCo-compatible core modules that are still imported by the
  renderer, such as `arguments/`, `gaussian_renderer/`, and `scene/`

## Current Problems

- Too many free-floating top-level scripts obscure the main render pipeline.
- Several shell wrappers duplicate the same renderer argument assembly with
  slightly different defaults.
- Smoke/debug utilities live beside production entry points.
- The `release/navdp_path_renderer` tree mirrors root code. That is useful as a
  deliverable, but it should be generated or tracked as an intentional package,
  not silently maintained by copying files.
- Data-transfer, storage, lighting, analysis, and rendering scripts are mixed
  together.
- Some removed legacy scripts had hard-coded local paths and saved debug arrays,
  which made the branch look less reproducible.

## Cleanup Already Done

- Removed generated/cache files: `.matplotlib-cache`, `__pycache__`,
  `.pytest_cache`, saved `.npy` debug arrays, and `parallel_render_report.json`.
- Removed historical local experiments:
  - gradio/manual/nav render demos with hard-coded local paths;
  - `render_label_paths_xiaotao.py`;
  - `run_datagen_xiaotao.sh`;
  - local notebook/debug result files and old GraphDeCo result assets.
- Removed one-off local operations scripts:
  - `move_subfolders.sh`, which encoded a fixed remote host and path;
  - `run_debug.sh`, which was a one-command debug invocation now covered by
    documented smoke/debug flags.
- Changed `gen_navdp_mask_ply.py` and `run_gen_navdp_mask_ply.sh` to default to
  repo-local `out/` output paths instead of fixed `/mnt` paths.
- Added ignore rules for Python/test caches, NumPy debug dumps, notebooks, and
  Matplotlib cache output.
- Removed the tracked private `scripts/.env` host config from Git, kept it
  ignored for local use, and added a sanitized `scripts/slave_finder.example.env`.

## Target Layout

Keep only high-level, user-facing entry points at repo root. Everything else
should live under a named package or `scripts/<domain>/`.

Recommended target:

```text
render_label_paths.py                 # compatibility wrapper
render_label_paths_telesim.py         # compatibility wrapper
parallel_render_paths.py              # compatibility wrapper
parallel_render_paths_telesim.py      # compatibility wrapper
run_random_fpv_datagen.sh             # compatibility wrapper
run_random_human_datagen.sh           # compatibility wrapper

navdp_datagen/
  render/
  runners/
  massgen/
  sensors/
  analysis/
  storage/

scripts/
  render/
  massgen/
  smoke/
  analysis/
  storage/
  media/
  lighting/
```

Long term, root wrappers should become thin calls into `navdp_datagen.*` modules
or into a single CLI.

## Script Classes

### Active User-Facing Entry Points

Keep these as stable wrappers until a unified CLI replaces them:

| Script | Role | Refactor Target |
| --- | --- | --- |
| `render_label_paths.py` | Legacy GS path renderer | Extract core renderer into `navdp_datagen.render.legacy` |
| `render_label_paths_telesim.py` | TeleSim path renderer | Extract core renderer into `navdp_datagen.render.telesim` |
| `parallel_render_paths.py` | Legacy parallel runner | Extract scheduler/run planning into `navdp_datagen.runners.parallel_legacy` |
| `parallel_render_paths_telesim.py` | TeleSim parallel runner | Extract scheduler/run planning into `navdp_datagen.runners.parallel_telesim` |
| `scripts/massgen/export_massgen_render_manifest.py` | MassGen scenario-to-render manifest export | Keep under `scripts/massgen/` after package extraction |
| `scripts/render/assets/render_glb_robot_overlay.py` | GLB robot compositor CLI | Keep under `scripts/render/` after package extraction |
| `scripts/smoke/run_g1_robot_follow_example.py` | G1 overlay example | Keep under `scripts/smoke/` or replace with documented smoke config |
| `scripts/datasets/gen_path_dataset_telesim.py` | Camera/actions dataset JSON export from TeleSim paths | Keep under `scripts/datasets/` and expose through MassGen/self-service launcher |
| `run_datagen.sh` | Legacy storage-aware renderer launcher | Keep temporarily; replace with config-driven launcher |

### Compatibility Shell Wrappers

These are useful today but should converge on one config-driven launcher:

| Script | Current Role | Cleanup Action |
| --- | --- | --- |
| `run_random_fpv_datagen.sh` | Main legacy FPV batch wrapper | Keep temporarily; migrate env defaults into a run config |
| `run_random_human_datagen.sh` | Main legacy follow-camera batch wrapper | Keep temporarily; migrate env defaults into a run config |
| `run_random_fpv_datagen_telesim.sh` | TeleSim FPV wrapper | Merge with FPV launcher config |
| `run_random_human_datagen_telesim.sh` | TeleSim follow-camera wrapper | Merge with follow launcher config |
| `gen_path_fpv_datagen_telesim.sh` | Path-generation plus FPV wrapper | Replace with pipeline subcommand |
| `gen_path_random_human_datagen_telesim.sh` | Path-generation plus follow wrapper | Replace with pipeline subcommand |
| `run_multihuman_demo_telesim.sh` | Multi-human demo wrapper | Move to `scripts/smoke/` if still needed |
| `run_waymo_fpv_datagen.sh` | Waymo-specific wrapper | Move to `scripts/render/waymo/` or remove if stale |
| `run_datagen.sh` | Storage-aware legacy renderer wrapper | Deprecate after unified launcher covers local/NAS/remote modes |

### Analysis and Reporting

Consolidate common path scanning, report writing, and plotting helpers:

| Script | Cleanup Action |
| --- | --- |
| `post_datagen_analysis.py` | Move logic into `navdp_datagen.analysis.post_run` |
| `datagen_analysis.py` | Merge overlapping report logic into `post_datagen_analysis.py` or package module |
| `actor_assignment_analysis.py` | Move under `scripts/analysis/` |
| `analyze_actor_sequences.py` | Move under `scripts/analysis/` |
| `analyze_selected_paths.py` | Move under `scripts/analysis/` |
| `scripts/chingmu_progress.py` | Keep as analysis utility; move to `scripts/analysis/` |
| `scripts/action_counts_report.py` | Move to `scripts/analysis/` |
| `scripts/compare_path_overlaps.py` | Move to `scripts/analysis/` |
| `scripts/diagnose_bottleneck.py` | Move to `scripts/analysis/` |
| `scripts/space_summary.py` | Move to `scripts/storage/` |
| `run_datagen_analysis.sh` | Replace with documented invocation or move to `scripts/analysis/` |
| `run_post_datagen_analysis.sh` | Replace with documented invocation or move to `scripts/analysis/` |
| `run_actor_assignment_analysis.sh` | Replace with documented invocation or move to `scripts/analysis/` |
| `analysis_utils.py` | Move reusable helpers into `navdp_datagen.analysis` |

### Smoke and Debug

Keep only smoke tests that are documented and run in CI/manual QA:

| Script | Cleanup Action |
| --- | --- |
| `scripts/smoke/quick_pipeline_test.py` | Keep under `scripts/smoke/` |
| `scripts/smoke/quick_gpu_pipeline_test.py` | Keep under `scripts/smoke/` |
| `scripts/quick_following_pipeline_test_telesim.py` | Keep only if it covers a distinct path; otherwise merge with quick GPU test |
| `run_quick_gpu_pipeline_test.sh` | Replace with documented invocation of `scripts/smoke/quick_gpu_pipeline_test.py` |
| `run_aa_side_by_side_test.py` / `.sh` | Move to `scripts/smoke/` or remove after anti-aliasing checks are covered elsewhere |
| `run_debug.sh` | Removed; debug commands should be examples in docs or smoke configs |
| `test_remote_storage.sh` | Move to `scripts/storage/` if still used; otherwise remove |
| `scripts/slave_finder.py` | Move to `scripts/smoke/` or remove if not part of server scheduling |

### Storage and Data Transfer

Keep these separate from render code:

| Script | Cleanup Action |
| --- | --- |
| `prepare_data_transfer.sh` | Move to `scripts/storage/` |
| `scripts/storage/storage_targets.sh` | Keep under `scripts/storage/`; root wrappers source this helper |
| `sync_human_sequences.sh` | Move to `scripts/storage/` |
| `parallel_copy_humanplys.sh` | Removed; superseded by configurable sync helpers |
| `move_subfolders.sh` | Removed; hard-coded local transfer helper |
| `scripts/storage/archive_depth_maps.sh` | Keep under `scripts/storage/` with explicit roots |
| `scripts/storage/cleanup_datagen_artifacts.sh` | Keep under `scripts/storage/` with explicit roots |

### Rendering Utilities

Group these by use case and remove overlaps after coverage exists:

| Script | Cleanup Action |
| --- | --- |
| `render_first_frame.py` | Keep as smoke/debug utility; move to `scripts/render/` |
| `render_room_center_views.py` | Move to `scripts/render/views/` |
| `render_room_topdown_views.py` | Move to `scripts/render/views/` |
| `run_room_center_views.sh` | Replace with config/invocation docs |
| `run_room_topdown_views.sh` | Replace with config/invocation docs |
| `batch_verify.py` | Move verification logic into `navdp_datagen.render.verify` |
| `bev_view.py` | Move reusable BEV rendering into `navdp_datagen.render.bev` |
| `reference_renderer_gpu.py` | Keep only if it is the canonical reference path; otherwise fold into tests |
| `gen_navdp_mask_ply.py` / `run_gen_navdp_mask_ply.sh` | Move to `scripts/render/assets/` |
| `scripts/render/compare/compare_camera_extrinsics.py` | Keep under `scripts/render/compare/` |
| `scripts/run_fpv_compare.sh` | Replace with documented compare command or move beside comparison tool |
| `scripts/render/render_multihuman_telesim.py` | Keep under `scripts/render/` or merge into TeleSim renderer CLI |
| `scripts/render/assets/convert_urdf_visuals_to_glb.py` | Keep under `scripts/render/assets/` |

### Dataset Conversion and Export

These are niche but may be valid; group them under explicit domains:

| Script | Cleanup Action |
| --- | --- |
| `scripts/export_frame_actions.py` | Move to `scripts/actions/` |
| `scripts/export_reverse_actions.py` | Move to `scripts/actions/` |
| `scripts/batch_add_actions_mp.py` | Move to `scripts/actions/` and extract shared multiprocessing helpers |
| `scripts/batch_add_betweenworld_actions_mp.py` | Move to `scripts/actions/` and extract shared multiprocessing helpers |
| `scripts/generate_dataset.py` | Move to `scripts/datasets/` |
| `scripts/generate_vlnpe_dataset.py` | Move to `scripts/datasets/` |
| `scripts/vlnpe2navllm.py` | Move to `scripts/datasets/` |
| `scripts/make_dataset_grids.py` | Move to `scripts/media/` |
| `scripts/make_random_video_mosaic.py` | Move to `scripts/media/` |
| `scripts/media/side_by_side_video_compare.py` | Keep under `scripts/media/` |
| `scripts/extract_video_frames.py` | Move to `scripts/media/` |
| `scripts/actions/generate_assignment_manifest.sh` | Keep under `scripts/actions/` until the unified CLI replaces it |
| `scripts/run_export_all.sh` | Replace with documented dataset export command |

### Legacy Training and Evaluation

These are not part of MassGen rendering, but they are still first-party or
GraphDeCo-compatible repo surface and should be classified instead of left
ambiguous:

| Script | Cleanup Action |
| --- | --- |
| `train.py` | Keep only if this repo still trains splats; otherwise move to `scripts/legacy_graphdeco/` or upstream docs |
| `render.py` | Keep only if still used for baseline Gaussian rendering; otherwise move to `scripts/legacy_graphdeco/` |
| `full_eval.py` | Move to `scripts/analysis/legacy_graphdeco/` or remove if stale |
| `metrics.py` | Move reusable metrics into package code or leave beside legacy eval |
| `convert.py` | Classify as conversion utility; move to `scripts/datasets/legacy_graphdeco/` if still used |
| `output_eval/generated_data_eval.py` | Move under `scripts/analysis/` or delete if no active output eval docs reference it |
| `utils/safe_room_view_point.py` | Keep test-covered room-view helper in `utils/` until package extraction |
| `scripts/actions/random_actor_assignments.py` | Keep until actor assignment is package-backed |

### Lighting

The lighting folder is already grouped. Keep it isolated, but add one README
section naming canonical commands and deprecate the rest:

| Script | Cleanup Action |
| --- | --- |
| `lighting/build_lighting_dataset.py` | Keep |
| `lighting/build_time_of_day_dataset.py` | Keep |
| `lighting/run_*` wrappers | Collapse into documented configs or one launcher |
| `lighting/*report*.py` | Consolidate shared report helpers |

### Release Mirror

`release/navdp_path_renderer` currently duplicates many root modules. It is
small, but it creates maintenance risk.

Plan:

1. Keep it for now because `docs/massgen_pathplanner_gsplat_handoff.md`
   references it as a deliverable.
2. Add a release packaging script that copies only the required source files
   from root into `release/navdp_path_renderer`.
3. Treat the release tree as generated output after that script exists, or move
   it to a separate package/repo.
4. Add a check that release files either match root sources or intentionally
   patch them.

## Refactor Order

1. Create `navdp_datagen.render` and extract pure camera/path/render helpers out
   of `render_label_paths.py` and `render_label_paths_telesim.py`.
2. Extract video writing, camera metadata, and path JSON output contracts into
   reusable modules.
3. Replace FPV/follow shell wrappers with a single config-driven launcher.
4. Move analysis scripts into `scripts/analysis/` and share report utilities.
5. Move storage scripts into `scripts/storage/` and replace shell-sourced config
   with explicit config files.
6. Move smoke/debug scripts into `scripts/smoke/` and delete any that are not
   documented or run.
7. Add the MassGen self-service launcher and sensor-rig profile support.
8. Generate or externalize `release/navdp_path_renderer`.
9. Move or archive legacy GraphDeCo train/render/eval entry points if MassGen no
   longer needs them at repo root.

## Deletion Rules

Delete a script when all are true:

- no active README/doc points users to it;
- no tests, run wrappers, or release scripts call it;
- it has hard-coded local paths, generated outputs, or one-off debug behavior;
- equivalent behavior exists in a current entry point or can be represented as a
  documented smoke command.

Move/refactor instead of deleting when:

- it is used by a current wrapper;
- it implements unique renderer, action, storage, or analysis behavior;
- it is referenced by Pathplanner handoff docs or platform scripts.

## Tracking Checklist

- [x] Remove generated report/cache/debug artifacts.
- [x] Remove known hard-coded local render experiments.
- [x] Remove one-off hard-coded transfer/debug scripts.
- [x] Convert mask-generation defaults from fixed `/mnt` paths to repo-local
  configurable paths.
- [x] Remove unreferenced hard-coded storage transfer scripts.
- [x] Make remaining storage cleanup/archive scripts require explicit roots.
- [x] Add `scripts/README.md` with domain ownership rules for future helpers.
- [x] Add package modules under `navdp_datagen/`.
- [x] Move top-level analysis scripts under `scripts/analysis/`.
- [x] Move storage/transfer scripts under `scripts/storage/`.
- [x] Move smoke/debug scripts under `scripts/smoke/` and document the survivors.
- [ ] Collapse FPV/follow shell wrappers into a config-driven launcher.
- [ ] Add a release packaging/check script for `release/navdp_path_renderer`.
- [ ] Delete or archive wrapper scripts after docs point to replacements.
