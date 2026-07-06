# Room View Safe Point Handoff

This is a handoff note for implementing safer room-center view generation on another machine.

## Current State

- The room-label downstream now supports BEV room overlays in `Code/room_label/vote_room_types.py`.
- This checkout has a room-center image generator:
  - `render_room_center_views.py`
  - `run_room_center_views.sh`
- It writes four horizontal RGB yaw images per room plus point-selection metadata:
  - `<output_dir>/<scene_id>/<room_number>_00.png`
  - `<output_dir>/<scene_id>/<room_number>_01.png`
  - `<output_dir>/<scene_id>/<room_number>_02.png`
  - `<output_dir>/<scene_id>/<room_number>_03.png`
  - `<output_dir>/<scene_id>/<room_number>.json`
  - `<output_dir>/<scene_id>/room_view_points.json`
- The safe-point selector is integrated into `render_room_center_views.py`.
  - It preserves the original center when valid.
  - It searches within `0.5m`, then `1.0m`, when the original point is invalid.
  - It marks rooms as `manual_verification_required` if no replacement is found.
  - Object bbox checks are height-aware when `camera_z` is supplied, so low desks/tables below the camera clearance do not force a camera move just because the XY footprint overlaps.
- A full live room-label test was run on all 12 `room_label_test/inputs/scenes` scenes with company_jdong.
- Output root from that run:
  - `room_label_test/outputs/company_jdong_bev_room_label_all`
- Each processed scene output contains:
  - `room_type_votes.json`
  - `room_type_overlay.png`
  - `room_bev_overlays/<room_id>_bev_overlay.png`
- The live test completed with:
  - `scenes=12`
  - `failures=0`
  - `rooms=61`
- Important behavior observed:
  - Each LLM request included 5 images: 4 RGB yaw views plus 1 BEV room overlay.
  - `gpt-5.5` frequently returned empty content from the UniAPI endpoint, but fallback `gpt-5.4` recovered those rooms.

## How To Run Room Image Generation

Run from the repo root:

```bash
cd /home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting
```

The wrapper defaults to:

- Conda env: `cuda121`
- Input scenes: `./data/scenes`
- Output dir: `./data2/<input-dir-name>_room_img`
- Camera height: `1.5m` above each room floor
- Image size: `512x512`
- Four yaw views per room

### Single Dataset

Generate room-center RGB images for the default `data/scenes` dataset:

```bash
./run_room_center_views.sh
```

This writes:

```text
data2/scenes_room_img/<scene_id>/
```

For a specific dataset and explicit output folder:

```bash
SCENES_DIR=./data/CHINGMU_rescaled_1 \
OUTPUT_DIR=./data2/CHINGMU_rescaled_1_room_img \
./run_room_center_views.sh
```

### Single Scene Smoke Test

Use `SCENE_ID` to render one scene before launching a full run:

```bash
SCENES_DIR=./data/scenes \
OUTPUT_DIR=./data2/scenes_room_img_smoke \
SCENE_ID=0905_840255 \
OVERWRITE=true \
./run_room_center_views.sh
```

### Batch Run For The Four Main Inputs

This is the batch command used for the June 18 run:

```bash
set -euo pipefail

for ds in CHINGMU_rescaled_1 CHINGMU_rescaled_2 CHINGMU_rescaled_3 scenes; do
  echo "[BATCH] Rendering $ds -> data2/${ds}_room_img"
  SCENES_DIR="./data/${ds}" \
  OUTPUT_DIR="./data2/${ds}_room_img" \
  ./run_room_center_views.sh
  echo "[BATCH] Finished $ds"
done
```

If replacing an existing run, rename old output folders first:

```bash
for ds in CHINGMU_rescaled_1 CHINGMU_rescaled_2 CHINGMU_rescaled_3 scenes; do
  if [ -d "./data2/${ds}_room_img" ]; then
    mv "./data2/${ds}_room_img" "./data2/${ds}_room_img_old"
  fi
done
```

Then run the batch command above.

### Useful Options

Set environment variables before the wrapper:

```bash
SCENES_DIR=./data/scenes          # input scene folder
OUTPUT_DIR=./data2/scenes_room_img # output folder
SCENE_ID=0905_840255              # optional single-scene filter
MAX_SCENES=10                     # optional first-N scene limit
OVERWRITE=true                    # rerender existing PNGs
CAMERA_HEIGHT=1.5                 # camera height above floor
WIDTH=512 HEIGHT=512              # image size
CONDA_ENV=cuda121                 # conda env used by conda run
USE_CONDA_RUN=true                # true/false/auto
```

### Outputs To Inspect

Each scene output folder contains:

```text
<room_number>_00.png
<room_number>_01.png
<room_number>_02.png
<room_number>_03.png
<room_number>.json
room_view_points.json
```

The metadata records whether the selected camera point changed:

```bash
jq '.rooms.room_02' data2/scenes_room_img/0905_840255/room_view_points.json
```

Relevant fields:

- `original_xy`: raw room center.
- `selected_xy`: actual camera XY used.
- `status`: `original_valid`, `adjusted_within_0.5m`, `adjusted_within_1.0m`, or `manual_verification_required`.
- `manual_verification_required`: true when no acceptable replacement was found.
- `reasons`: why the original point failed.
- `collided_label_ids` / `collided_structure_ids`: object or structure collisions detected for the original point.

Quick summary commands:

```bash
for ds in CHINGMU_rescaled_1 CHINGMU_rescaled_2 CHINGMU_rescaled_3 scenes; do
  echo "$ds"
  find -L "data2/${ds}_room_img" -name room_view_points.json -print0 \
    | xargs -0 jq -r '.rooms[]?.status' \
    | sort | uniq -c | sort -nr
done
```

Count outputs:

```bash
for ds in CHINGMU_rescaled_1 CHINGMU_rescaled_2 CHINGMU_rescaled_3 scenes; do
  echo "$ds scene dirs: $(find -L data2/${ds}_room_img -mindepth 1 -maxdepth 1 -type d | wc -l)"
  echo "$ds metadata:   $(find -L data2/${ds}_room_img -name room_view_points.json | wc -l)"
  echo "$ds png files:  $(find -L data2/${ds}_room_img -name '*.png' | wc -l)"
done
```

### Known Input Caveat

`data/scenes` may contain non-scene directories. In the June 18 run:

- `data/scenes/1002_839955` only had `wall_mask.json` and `wall_mask.png`.
- `data/scenes/__pycache__` was a Python cache directory.

Those entries caused `failed_scenes=2` after all valid scenes rendered. If needed, use `SCENE_ID` or clean/filter the input directory before a strict batch run.

### Archive Command

To archive the generated image folders, run:

```bash
tar -cf ./data2/room_img_Jun18.tar -C ./data2 \
  CHINGMU_rescaled_1_room_img \
  CHINGMU_rescaled_2_room_img \
  CHINGMU_rescaled_3_room_img \
  scenes_room_img
```

## Problem To Fix

The existing four-view room RGB image generation may pick a camera point that is:

- Inside a wall or black/non-free area in `occupancy.png`.
- Inside an object footprint from `labels.json` bounding boxes.
- Horizontally overlapping a wall, window, or door in `structure.json`.
- Very close to geometry or clutter, making the four rendered yaw images poor evidence for room-type labeling.

If the picked point is invalid, the renderer should move it to the nearest acceptable free point:

1. First search within `0.5` meters horizontally.
2. If no valid point is found, search within `1.0` meter horizontally.
3. If still no valid point is found, use the original point but mark that room as requiring manual verification.

The downstream room-label pipeline must accept and preserve this manual-check label.

## Renderer Location Status

This checkout now has the four-view 3D renderer script that writes:

```text
data2/<dataset>_room_img/<scene_id>/<room_number>_00.png
data2/<dataset>_room_img/<scene_id>/<room_number>_01.png
data2/<dataset>_room_img/<scene_id>/<room_number>_02.png
data2/<dataset>_room_img/<scene_id>/<room_number>_03.png
```

Relevant files:

- `render_room_center_views.py`
  - Selects room center/safe point.
  - Renders four yaw images per room.
  - Writes per-room JSON and `room_view_points.json`.
- `safe_room_view_point.py`
  - Implements safe-point validation/search.
- `run_room_center_views.sh`
  - Wrapper for the center-view renderer.
- `render_room_topdown_views.py` and `run_room_topdown_views.sh`
  - Separate top-down/BEV render path.

## Implementation Reference

Reusable safe-point logic is implemented and called from the four-view renderer before rendering the four yaw images.

Current module:

```text
safe_room_view_point.py
```

Current public API:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

@dataclass(frozen=True)
class SafeRoomViewPoint:
    original_xy: tuple[float, float]
    selected_xy: tuple[float, float]
    selected_pixel: tuple[int, int]
    status: str
    manual_verification_required: bool
    search_radius_m: Optional[float]
    reasons: list[str]
    collided_label_ids: list[str]
    collided_structure_ids: list[str]

def choose_safe_room_view_point(
    scene_dir: Path,
    original_xy: tuple[float, float],
    *,
    room_polygon: Optional[Sequence[Sequence[float]]] = None,
    occupancy_threshold: int = 200,
    occupancy_clearance_m: float = 0.10,
    structure_margin_m: float = 0.08,
    object_margin_m: float = 0.03,
    camera_z: float | None = None,
    object_vertical_clearance_m: float = 0.20,
    search_radii_m: Sequence[float] = (0.5, 1.0),
) -> SafeRoomViewPoint:
    ...
```

Status values:

```text
original_valid
adjusted_within_0.5m
adjusted_within_1.0m
manual_verification_required
```

## Validation Rules

A candidate point is valid only if all checks pass.

### 1. Occupancy Check

- Load `occupancy.png` as grayscale.
- Convert world XY to occupancy pixel using the repo's left-handed coordinate conversion:
  - `Code/navdp/coords.py`
  - `world_to_pixel_left_handed(...)`
  - `pixel_to_world_left_handed(...)`
- Reject if outside image bounds.
- Reject if pixel value is below `occupancy_threshold` (default `200`).
- Strongly prefer applying a small clearance around the point, not just the exact pixel:
  - Recommended clearance: `0.08m` to `0.15m`, converted to pixels.
  - Reject if any pixel in that clearance disk is non-free.

### 2. Object Bounding Box Footprint Check

- Load `labels.json`.
- For each label entry with `bounding_box`, compute horizontal footprint:
  - `min_x`, `max_x`, `min_y`, `max_y` from bbox vertices.
- Also preserve object height when bbox vertices include `z`:
  - `min_z`, `max_z`.
- Reject a candidate on object collision if:
  - `min_x - object_margin_m <= x <= max_x + object_margin_m`
  - `min_y - object_margin_m <= y <= max_y + object_margin_m`
  - and the object is not clearly below the camera.
- If `camera_z` is supplied and `max_z + object_vertical_clearance_m < camera_z`, treat that object as below the camera and do not reject solely because the XY footprint overlaps.
- If the occupancy pixel is non-free only because of a low object below the camera, allow the original point; walls/doors/windows and tall objects still reject.
- Record the collided object IDs in `collided_label_ids`.
  - Prefer `ins_id`; fall back to `label` plus list index.

### 3. Structure Wall/Window/Door Check

- Load `structure.json`.
- Reject candidates horizontally overlapping these structure elements:
  - walls
  - windows
  - doors
- Structure formats vary, so implement tolerant extraction:
  - Segment form: `[x1, y1, x2, y2]`
  - Dict endpoints: `{"start": [...], "end": [...]}`
  - Dict points: `{"points": [[x, y], ...]}`
  - Bbox/profile-like polygon lists when present
- For line segments, reject when point-to-segment distance <= `structure_margin_m`.
- For polygons/bboxes, reject if point is inside or within margin.
- Record collided structure IDs/types in `collided_structure_ids`.

### 4. Room Polygon Constraint

If the renderer knows the room polygon, keep adjusted points inside the room polygon.

- Use the room `profile` from `structure.json` if available.
- If a room polygon is not available, still run occupancy/object/structure checks.
- Do not move the point into another room if room polygon data is available.

## Nearest Free Point Search

If the original point is invalid:

1. Convert original XY to occupancy pixel.
2. Search candidate pixels in increasing distance order.
3. First cap distance at `0.5m`.
4. If no candidate passes, repeat with `1.0m`.
5. Convert accepted pixel back to world XY.
6. If no candidate passes either radius:
   - Return original XY.
   - Set `manual_verification_required=True`.
   - Set status to `manual_verification_required`.

Efficient implementation:

- Convert radius meters to pixels using occupancy scale.
- Iterate rings or collect candidate pixels within radius and sort by squared distance.
- Only test pixels whose occupancy is free first, then run the slower object/structure checks.

## Metadata To Write Beside Rendered Room Images

The renderer should write a metadata file beside the four PNGs.

Recommended file:

```text
room_label_test/inputs/scene_imgs/<source>/<scene_id>/room_view_points.json
```

Recommended schema:

```json
{
  "scene_id": "0689_841515",
  "source": "InteriorGS",
  "rooms": {
    "room_01": {
      "room_number": 1,
      "original_xy": [1.23, 4.56],
      "selected_xy": [1.34, 4.50],
      "selected_pixel": [123, 456],
      "status": "adjusted_within_0.5m",
      "manual_verification_required": false,
      "search_radius_m": 0.5,
      "reasons": ["original_overlaps_object_bbox"],
      "collided_label_ids": ["chair_12"],
      "collided_structure_ids": []
    }
  }
}
```

If the renderer already writes per-room metadata, extend it instead of adding a new file.

## Downstream Changes Required

Update `Code/room_label/vote_room_types.py` so it accepts and preserves the manual-check metadata.

### Add Metadata Loading

When resolving room images in `resolve_room_image_paths(...)`, also look for:

```text
<room_dir>/room_view_points.json
```

Map by `room_id`, e.g. `room_01`, not by raw image filename.

### Extend SceneRun

Current `SceneRun` already has:

```python
image_inputs_by_room: Dict[str, List[str]]
```

Add:

```python
room_view_metadata_by_room: Dict[str, Dict[str, Any]]
```

or similar.

### Preserve In Outputs

In `finalize_scene_votes(...)`, add each room's view metadata:

```json
{
  "room_id": "room_01",
  "image_inputs": [...],
  "view_point": {
    "status": "manual_verification_required",
    "manual_verification_required": true,
    "original_xy": [...],
    "selected_xy": [...],
    "reasons": [...]
  },
  "manual_verification_required": true,
  "final_room_type": "corridor"
}
```

Also add a scene-level count in `room_type_vote_summary.json`, for example:

```json
"manual_verification_room_count": 3
```

### Prompt Behavior

If a room has `manual_verification_required=True`, do not block the LLM call. Instead:

- Keep the four room images if they exist.
- Keep the BEV overlay.
- Add a short warning to the user prompt:

```text
Note: the room-center RGB camera point failed safe-point checks and is marked for manual verification. Use the BEV overlay and object labels more heavily if the RGB views look obstructed.
```

This lets the pipeline continue while making the uncertainty visible in saved artifacts.

### Visualization Behavior

Update the scene-level `room_type_overlay.png` renderer in `vote_room_types.py`:

- If `manual_verification_required=True`, draw a visible marker on that room label.
- Suggested label text:
  - `3: corridor [check]`
- Suggested color:
  - keep room fill color unchanged
  - draw label outline or small warning ring in red/orange

This is important because the user asked for downstream handling, not just metadata storage.

## Tests To Add

Add focused unit tests. Suggested file:

```text
Code/room_label/test_safe_room_view_point.py
```

Test cases:

1. `test_original_valid_point_is_kept`
   - Create tiny occupancy with white free space.
   - Candidate not inside labels bbox or structure segment.
   - Expect `status == "original_valid"`.

2. `test_point_inside_object_moves_to_nearest_free_point`
   - Candidate inside a label bbox.
   - Nearby free point exists within `0.5m`.
   - Expect selected point differs and `manual_verification_required is False`.

3. `test_point_on_black_occupancy_moves_to_free_point`
   - Candidate pixel is black.
   - Nearby white pixel exists.
   - Expect adjusted status.

4. `test_point_on_wall_or_door_moves`
   - Add a wall/door/window segment in `structure.json`.
   - Candidate lies within margin.
   - Expect adjusted status and structure collision reason.

5. `test_unfixable_point_requires_manual_verification`
   - Block all candidates within `1.0m` or make all candidates collide.
   - Expect original point is returned and `manual_verification_required is True`.

6. `test_vote_room_types_preserves_manual_check_metadata`
   - Build fake `room_view_points.json`.
   - Run the relevant prepare/finalize path without LLM if possible, or directly test metadata loader.
   - Confirm `room_type_votes.json` would include `manual_verification_required`.

Compile/test commands:

```bash
python3 -m py_compile Code/room_label/vote_room_types.py Code/room_label/safe_room_view_point.py
python3 -m pytest Code/room_label/test_safe_room_view_point.py
```

## Smoke Test After Implementation

First regenerate the four-view images on the target machine using the patched renderer.

Then run an audit-only room label pass to verify metadata/overlays without model spend:

```bash
python3 Code/room_label/vote_room_types.py \
  --scenes-root room_label_test/inputs/scenes \
  --room-inputs-root room_label_test/inputs/scene_imgs \
  --outputs-root room_label_test/outputs/safe_point_audit \
  --audit-only \
  --allow-low-coverage
```

Then run a single-scene live smoke:

```bash
python3 Code/room_label/vote_room_types.py \
  --scenes-root room_label_test/inputs/scenes \
  --room-inputs-root room_label_test/inputs/scene_imgs \
  --outputs-root room_label_test/outputs/company_jdong_safe_point_smoke \
  --allow-low-coverage \
  --save-structure-variants \
  --scene 0029_858861 \
  --max-retries 1 \
  --expert 'name=company_jdong,provider=openai,model=gpt-5.5,fallback_models=gpt-5.4,api_key_env=JDONG_COMPANY_KEY,base_url=https://api.uniapi.io/v1'
```

Finally run all scenes if the smoke passes:

```bash
python3 Code/room_label/vote_room_types.py \
  --scenes-root room_label_test/inputs/scenes \
  --room-inputs-root room_label_test/inputs/scene_imgs \
  --outputs-root room_label_test/outputs/company_jdong_safe_point_all \
  --allow-low-coverage \
  --save-structure-variants \
  --max-retries 1 \
  --expert 'name=company_jdong,provider=openai,model=gpt-5.5,fallback_models=gpt-5.4,api_key_env=JDONG_COMPANY_KEY,base_url=https://api.uniapi.io/v1'
```

## Manual Review Checklist

For each scene output, inspect:

- `room_type_overlay.png`
  - Room types are visible.
  - Rooms marked `[check]` are easy to find.
- `room_bev_overlays/<room_id>_bev_overlay.png`
  - Target room is highlighted.
- `room_type_votes.json`
  - Each room has `image_inputs`.
  - Each room has `view_point` if `room_view_points.json` existed.
  - Manual-check rooms have `manual_verification_required: true`.
- `room_type_vote_summary.json`
  - Aggregates manual-check room counts.

## Notes For The Implementing Codex

- Do not remove the BEV overlay changes in `vote_room_types.py`.
- Keep the implementation tolerant of missing `room_view_points.json`; old image folders must still work.
- Do not fail the room-label run just because a room requires manual verification.
- Treat manual verification as a saved warning/annotation, not a hard error.
- Keep edits scoped: point selection utility, renderer integration, downstream metadata handling, and tests.
