# Jiankun Change Log (Walking Sequence Feature)

## Completed
- Added CLI switches (`--actor-seq-dir`, `--actor-pattern`, `--actor-height`, `--actor-speed`, `--actor-fps`, `--follow-distance`, `--follow-buffer`, `--actor-foot-offset`, `--actor-no-loop`) to gate the animated walker workflow.
- Updated the default actor glob to `*.ply` with natural sorting so numeric frame names load automatically without extra flags.
- Implemented actor sequence loading/normalisation (`load_actor_sequence`) with axis alignment (up = `-Y`, forward = `+Z`), uniform scaling to 1.7 m, and ground-plane rebaselining.
- Added `utils/gaussian_ply_utils.py` to apply rigid transforms while preserving Gaussian attributes (including SH coefficients), and reused it for per-frame placement along the path (actor PLYs must be valid, no GPU fallback).
- Created `render_actor_follow_sequence` to step the actor along the polyline at constant speed, loop animation frames, and drive the camera trailing behaviour with start/end easing.
- Introduced `CombinedGaussianModel` to merge the static scene model with per-frame actor gaussians without mutating the base tensors.
- Updated `render_path_frames` main loop to branch into the actor rendering path while retaining legacy behaviour when no actor options are provided.
- Documented implementation details in `Jiankun_Design.md`.
- Added in-path scene cropping so only gaussians near the actor/camera window are duplicated, keeping GPU memory requirements manageable during actor renders.

## Notes
- Camera follow currently requires `--view-mode forward`; top-down support is deferred.
- Catch-up duration is tied to actor speed to keep motion natural; adjust `--actor-speed` for snappier finales if needed.
