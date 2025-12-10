# Walking Sequence Rendering Design

## Goal
- Extend `render_label_paths.py` to support rendering animated 3D Gaussian splats of a person walking along navigation paths.
- Loop a sequence of per-frame `.ply` files to animate the person as they traverse the path.
- Drive a following camera that stays on the path, approximately 1.5 m behind the person while in motion, and transitions to 0.75 m separation at the start and end.

## High-Level Flow
- Load the base static scene Gaussian model (existing behavior).
- Load a configurable sequence of actor `.ply` files representing animation frames; reuse or stream into GPU memory.
- For each sampled point along the navigation path:
  - Determine actor position/orientation aligned to path tangent.
  - Select animation frame (looping).
  - Apply transform to actor Gaussian points to place the actor in world coordinates.
  - Place camera on path with trailing offset constraints; update orientation to look at actor hips/torso.
  - Render combined scene (static + actor) for the frame.
- Terminate once the actor reaches the path end. If camera stopped short due to trailing distance, advance it forward until 0.75 m from actor end position while keeping actor static on final frame(s).

## CLI Additions (draft)
- `--actor-seq-dir PATH`: directory containing ordered per-frame `.ply` files for the animated actor.
- `--actor-pattern GLOB`: optional glob to filter sequence files (default: `[0-9][0-9][0-9][0-9][0-9][0-9].ply`).
- `--actor-height FLOAT`: approximate actor height for scaling (default: `1.7`).
- `--follow-distance FLOAT`: preferred trailing distance during motion (default: `1.5` meters).
- `--follow-buffer FLOAT`: start/end separation (default: `0.75` meters).
- `--actor-fps FLOAT`: playback rate for the actor animation (default: derived from path sampling or `DEFAULT_VIDEO_FPS`).
- `--actor-loop`: enable looping of actor frames (default: true; new flag to disable looping and clamp).

_Note_: Flags will be ignored when `--actor-seq-dir` is not provided to maintain existing behavior.

## Data Handling
- Enumerate `.ply` files under `--actor-seq-dir` using the provided glob, ordered by filename. Validate non-empty sequence.
- Load each frame with `utils.gaussian_ply_utils.GaussianPly`, apply the SuperSplat axis alignment, and compute a shared uniform scale so the tallest pose matches `--actor-height`.
- Translate all frames so the global minimum `z` rests at zero, caching the processed structured arrays for reuse.
- During rendering, clone the cached frame, apply the per-step rigid transform via `apply_transform_inplace`, and convert the transformed attributes to torch tensors before rasterisation.

## Path Parameterization
- Reuse existing `sampled_xy` path but compute cumulative arc length to support distance-based placement.
- Derive actor positions by walking the path using actor stride distance:
  - Convert navigation polyline into evenly spaced samples with spacing tied to desired walking speed (e.g., 1.3 m/s) and frame rate.
  - Maintain mapping from frame index to path distance.
- Loop actor animation frames: `frame_idx % len(actor_frames)`.
- Align actor orientation using forward direction vector (already available) and align yaw so actor faces `forward`.

## Actor Transform
- Build rigid transform per frame:
  - Translation: path XY at current distance; Z: align feet to ground using occupancy lower bound plus actor foot offset.
  - Rotation: align actor local forward axis with path tangent around vertical axis.
    - Actor asset coordinates: up axis is `-Y`, forward axis is `+Z` when viewed in SuperSplat. Convert by swapping axes so world-up (`+Z` in renderer) matches actor `-Y`, and world-forward (path tangent in XY plane) aligns with actor `+Z`.
  - Optional scaling: if actor sequence is not 1.7 m tall, compute uniform scale factor using bounding box height vs `--actor-height`.
- Apply transform by modifying actor Gaussian `_xyz` before rendering. Avoid permanent mutation by storing base positions and reapplying each frame.

## Camera Logic
- Compute camera position along path at `actor_distance - follow_distance`, clamped within path domain.
- Blend trailing distance from `follow_buffer` to `follow_distance` over initial `N` frames to avoid abrupt start; similar blend when approaching path end.
- When actor reaches final target, keep actor static for optional hold frames while camera closes gap to `follow_buffer`.
- Camera orientation:
  - Look target: actor hip joint (position + `actor_height * 0.6`).
  - Up vector: world Z.
  - Continue to support `--look-down`, `--look-ahead` overrides when not specifying actor.

## Rendering Integration
- Extend `render_path_frames` to branch into `render_actor_walk_sequence` when actor options provided.
- Reuse existing video/frame writing logic.
- Manage GaussianRenderer input by combining static and dynamic gaussians:
  - Option A: instantiate second `GaussianModel` for actor and adapt `render_or` to accept list of models (preferred if renderer already supports batching).
  - Option B: temporarily concatenate actor gaussians into scene model tensors each frame (need to clone/restore to avoid data races).
- Ensure torch device operations avoid reallocations; consider caching actor tensors on GPU.

## Edge Cases & Validation
- Path length shorter than follow distances: clamp camera start/end positions and adjust blending gracefully.
- Missing actor assets: emit clear error and skip rendering.
- Actor ground plane mismatch: expose `--actor-foot-offset` to manually tweak vertical alignment.
- Performance: measure memory usage when caching all actor frames; add warning if exceeds threshold.

## Usage Example
```bash
python render_label_paths.py \
  --scene outdoor_plaza \
  --label-id 12 \
  --actor-seq-dir data/actors/walker_A \
  --actor-pattern "walker_frame_*.ply" \
  --actor-height 1.7 \
  --follow-distance 1.5 \
  --follow-buffer 0.75 \
  --video
```
- Produces frames/video where the animated walker traverses the label path.
- Camera trails at ~1.5 m, easing in/out to 0.75 m at endpoints.

## Open Questions
- Confirm availability of pre-aligned actor `.ply` (local forward axis, origin).
- Decide on target walking speed or reuse path sampling stride.
- Determine whether to add multi-actor support in future revisions.

## Implementation Notes
- Added `render_actor_follow_sequence` that orchestrates actor placement and camera motion along the sampled polyline. The helper builds a `PathSampler` for distance-to-XY queries, drives animation timing, and appends end-of-path catch-up frames so the camera finishes at the requested buffer distance.
- Introduced `CombinedGaussianModel` to merge static scene tensors with the transient actor frame without mutating the original `GaussianModel`. This wrapper exposes the minimal `get_*` interface required by `render_or`.
- New CLI flags (`--actor-seq-dir`, `--actor-pattern`, `--actor-height`, `--actor-speed`, `--actor-fps`, `--follow-distance`, `--follow-buffer`, `--actor-foot-offset`, `--actor-no-loop`) enable the walking overlay. Flags are ignored unless `--actor-seq-dir` is provided.
- Actor `.ply` frames are normalised with `gaussian_ply_utils.apply_transform_inplace`, rotating from (up: `-Y`, forward: `+Z`) into world coordinates, scaling to the requested 1.7 m height, and re-basing the ground plane to `z = 0`; malformed inputs now surface parsing errors immediately.
- Per-frame transforms clone the cached structured array, apply the path-aligned rigid transform through the same utility (preserving normals, scales, quaternions, and SH coefficients), and translate the character so feet sit at `floor_z + actor_foot_offset`.
- A reusable `CombinedGaussianModel` preallocates the scene + actor tensors once on the GPU and updates only the actor slice each frame, eliminating the per-frame duplication that previously caused large CUDA allocations.
- For memory safety we crop the static scene to an axis-aligned window around the actor/camera path (configurable margin), so only nearby gaussians are copied before compositing with the actor.
- Camera positions project onto the same polyline but with trailing distance easing from buffer → follow distance during motion, then a catch-up loop brings the camera back to the buffer spacing after the actor stops.
