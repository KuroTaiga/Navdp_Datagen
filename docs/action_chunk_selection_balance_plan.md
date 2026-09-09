# Datagen Frame-Interest Selection And Balance Plan

Status: implementation-supported design
Repo: `Navdp_Datagen`
Scope: select model-useful navigation frames before expensive rendering, expand
each selected frame into its required temporal context, and pass only selected
windows to the renderer.

## Purpose

This selector sits between Pathplanner scenario generation and
`Navdp_Datagen` rendering. Pathplanner still owns mission/scenario construction
and path-quality filtering. Datagen now owns a second filtering step: choose
which frames are informative enough for train/test examples, then render exactly
the temporal context required by those choices.

The model contract for each selected frame of interest is:

```json
{
  "past_context_frame_count": 32,
  "current_frame_count": 1,
  "future_context_frame_count": 32,
  "render_window_frame_count": 65
}
```

If frame `t` is selected for training or testing, the renderer must preserve and
render all frames `t-32..t+32`. At a required path endpoint, the nearest edge
sample is repeated so the endpoint remains at index 32. Stationary, yield,
edge-padded, and repeated-pose frames are valid context and must not be
deduplicated away.

## Research Basis

The attached Nature D2RL paper argues for removing non-critical, uninformative
states and densifying the learner's exposure to safety-critical states. The
same principle is useful here, but the navigation task needs a deterministic
dataset selector rather than an online RL simulator edit. Fixed action ratios
such as 10/15/15/60 or 15/15/15/55 are therefore treated as starting heuristics,
not as an optimal distribution.

For NavDP-style navigation training, the selector uses two levels of guidance:

1. Primary interest buckets decide which planning moments are worth rendering.
2. Secondary action ratios provide a soft tie-breaker and survey reference.
   They are not quotas and do not reject an otherwise valuable point.

Primary references:

- D2RL safety-critical state densification:
  https://www.nature.com/articles/s41586-023-05732-2
- DAgger distribution shift framing for sequential imitation learning:
  https://proceedings.mlr.press/v15/ross11a.html
- NavDP sim-to-real navigation diffusion policy and large-scale simulated data:
  https://arxiv.org/abs/2505.08712

## Default Distribution Policy

Implemented policy: `anchor_aware_center_balance/v0.3`.

Primary interest buckets:

| Bucket | Meaning |
| --- | --- |
| `critical_margin` | low clearance, near-collision margin, safety event, or near actor/robot |
| `human_interaction` | target/bystander human is close enough to affect the path |
| `route_decision` | turn apex, stop/yield transition, release/deadline/event proximity |
| `representative_motion` | clean motion needed for coverage after the informative states are sampled |

Default bucket priors favor critical and interaction states, then route
decisions. Representative motion remains a first-class interest bucket. The
actual target ratios are derived from the available candidate distribution per
run, and operators can override them with `--bucket-ratio`. These priors guide
which scored points are selected; they are not hard caps on action types.

Secondary action target ratios default to:

| Action | Code | Default Share |
| --- | ---: | ---: |
| stop | `0` | 15% |
| forward / move | `1` | 55% |
| turn left | `2` | 15% |
| turn right | `3` | 15% |

These ratios are configurable with `--action-ratio`, but currently affect only
soft selection priority. The survey reports action distributions at selected
centers and across the complete retained 33-frame windows. This avoids mistaking
a critical-state-heavy center distribution for a dataset with no forward motion.

## Inputs

The implemented selector reads one or more MassGen render manifests generated
from Pathplanner scenarios. It uses:

- `jobs[*].camera.trajectory` for frame, time, pose, yaw, and motion state;
- mission release/deadline times and event logs for route-decision salience;
- manifest human trajectories and peer-robot pose tracks for proximity scores;
- job family and actor ids for chunk metadata and downstream filtering.

Future Pathplanner candidate-quality metadata can be added as another feature
source without changing the selected-window render contract.

## Candidate Scoring

The sampler first chooses complete source paths by deterministic seeded random
rank, then scores every frame on each chosen path. It never feeds a pre-cut path
fragment into candidate scoring. `--max-source-paths 0` selects all eligible
paths; a positive value selects that many whole paths.

Every camera trajectory sample that can support the requested window becomes a
candidate. The default edge policy is `reject`, so an ordinary point requires 32
valid past frames. Required sub-mission and mission endpoints use the configured
endpoint policy, `clamp` by default. No future frames are packaged.

Each candidate stores:

- source manifest index/path and source render job id;
- source frame index/id, timestamp, pose, yaw, and coarse action;
- all 33 source frame indices required for rendering;
- interest bucket scores and selection reasons;
- split and target role metadata.

Pathplanner trajectories may contain sparse waypoints whose integer `frame`
values span many renderer frames. By default, the selector linearly interpolates
position, time, and wrapped yaw across those frame-id gaps before applying the
33-frame eligibility rule. It preserves source/interpolation provenance in point
metadata. `--no-densify-frame-gaps` is available only for compatibility audits.

Score components include actor/robot proximity, event proximity, stop/turn
transitions, turn magnitude, and representative clean-motion coverage.

## Selection Algorithm

1. Load MassGen render manifests and apply mission-family filtering.
2. Deterministically random-rank complete source paths using the run seed, then
   retain all paths or the requested `--max-source-paths` subset.
3. Densify each selected path and score every eligible frame on the full path.
4. Mark assigned sub-mission completion/checkpoint frames and each trajectory end
   as mandatory anchors.
5. Derive adaptive interest-bucket targets for the requested scored POI budget.
6. Add mandatory anchors first. They are additive and do not consume the scored
   POI budget, per-path POI cap, or action-deficit counts.
7. Select scored POIs by bucket score, soft action deficit, deterministic seed
   jitter, and minimum spacing. Multiple POIs may come from the same path;
   `--max-targets-per-job 0` leaves that unlimited.
8. Fill any remaining scored POI budget from the globally best eligible frames.
9. Emit 33-frame windows and both selection and selected-render manifests.

The manifest separately reports actions at all selected centers, scored POI
centers, mandatory anchors, all repeated window samples, and unique retained
source frames. Availability-aware action minimums apply only to scored POIs;
mandatory mission and trajectory endpoints remain in a separate additive pool.
No per-path action maximum is imposed.

## Selection Manifest

`scripts/massgen/select_frame_interest_windows.py` writes
`navdp_frame_interest_selection/v0.1` manifests:

```json
{
  "schema_version": "navdp_frame_interest_selection/v0.1",
  "config": {
    "past_frames": 32,
    "future_frames": 0,
    "window_frame_count": 33,
    "distribution_policy": "anchor_aware_center_balance/v0.3"
  },
  "distribution": {
    "target_bucket_ratios": {
      "critical_margin": 0.4,
      "human_interaction": 0.25,
      "representative_motion": 0.1,
      "route_decision": 0.25
    },
    "target_action_ratios": {
      "move": 0.55,
      "stop": 0.15,
      "turn_left": 0.15,
      "turn_right": 0.15
    }
  },
  "chunks": [
    {
      "target_role": "frame_of_interest",
      "target_frame": 120,
      "target_action_name": "turn_left",
      "target_bucket": "route_decision",
      "window": {
        "source_frame_indices": [88, 89, 90],
        "render_frame_count": 65
      },
      "render_contract": {
        "must_render_all_window_frames": true,
        "preserve_stationary_frames": true
      }
    }
  ]
}
```

The sample shortens `source_frame_indices` for readability. Real manifests write
all 65 indices.

## CLI

Write a selection manifest:

```bash
python3 scripts/massgen/select_frame_interest_windows.py \
  --manifest-json out/massgen/render_manifest.json \
  --family dense_dynamic_avoidance \
  --max-source-paths 20 \
  --target-count 1000 \
  --output-selection-json out/frame_selection/selection_manifest.json
```

Prepare the all-family pilot matrix before source manifests are connected:

```bash
python3 scripts/massgen/prepare_frame_selection_family_pilot.py \
  --output-root out/frame_selection_family_pilot
```

This writes a non-executing `pilot_plan.json` with every active family marked
`waiting_for_source_manifest`. Once local or mounted databank manifests are
available, prepare selections and self-contained selected render manifests with:

```bash
python3 scripts/massgen/prepare_frame_selection_family_pilot.py \
  --scenario-json /path/to/pathplanner_scenario.json \
  --output-root out/frame_selection_family_pilot \
  --targets-per-family 2 \
  --render-repo-root /team/telenav/code/Navdp_Datagen \
  --remote-sources-connected
```

Both repeatable `--scenario-json` and `--manifest-json` inputs are supported.
Scenario inputs are converted into renderer-owned manifests under
`source_render_manifests/` before family selection.

The pilot generator never runs render commands. Each ready family record stores
separate `plan_argv` and `execute_argv` arrays, both with
`execution_authorized=false`, so operators can inspect render plans before
explicitly launching the 33-frame jobs.

The default pilot also includes four separately audited social-navigation
variants mapped to their law ids: personal space (`L1`), pedestrian yield
(`L2`), group integrity (`L3`), and queue order (`L4`). Dedicated single-family
manifests are preferred over mixed `mission_stream` manifests; mixed sources are
used only when a family has no dedicated source.

For an unambiguous family comparison, source exports should contain one mission
family per manifest or accurate job-level family bindings. If every job in a
multi-family manifest carries every family label, the same trajectory can enter
more than one family pilot; the pilot plan preserves that fact rather than
guessing which mission produced the frames.

Generate BEV overlays and distribution plots from a completed pilot without
running Gaussian rendering:

```bash
python3 scripts/massgen/visualize_frame_selection_outputs.py \
  --pilot-root out/frame_selection_family_pilot \
  --source-manifest-root out/frame_selection_family_pilot/source_render_manifests \
  --output-root out/frame_sampling_visualization
```

Write both selection and selected-window render manifest:

```bash
python3 scripts/massgen/select_frame_interest_windows.py \
  --manifest-json out/massgen/render_manifest.json \
  --target-count 1000 \
  --output-selection-json out/frame_selection/selection_manifest.json \
  --output-render-manifest-json out/frame_selection/selected_render_manifest.json
```

Plan/render directly from a source render manifest and selection manifest:

```bash
python3 scripts/massgen/render_manifest_jobs.py \
  --manifest-json out/massgen/render_manifest.json \
  --frame-selection-json out/frame_selection/selection_manifest.json \
  --write-inputs
```

## Renderer Integration

Current implementation prepares one selected-window job per selected target. Each
window job contains a 33-point camera trajectory, rewrites renderer-local frame
ids to `0..32`, and preserves original source frame indices in metadata.

The renderer path now respects `camera.preserve_frame_samples` and
`metadata.preserve_frame_samples`, so repeated stationary poses are kept in the
label path and are not removed by path deduplication or distance resampling.
Executor planning also blocks `--minimal-frames` values that would truncate a
selected frame-interest window.

The selection summary reports both:

- `requested_window_frame_renders`: selected targets multiplied by 33;
- `unique_source_render_frame_count`: de-duplicated source frames needed across
overlapping windows.

Selections can be constrained with repeatable `--family` arguments. The
all-family pilot generator uses that constraint to prevent multi-family source
collections from filling one family's budget with another family's jobs.

A future sparse-frame renderer can render each unique source frame once and map
it back to every chunk. Until then, the selected-window manifests preserve the
33-frame model contract without authorizing or executing rendering.

## Output Layout

Suggested local layout:

```text
out/frame_interest_selection/<run_id>/
  selection_manifest.json
  selected_render_manifest.json
  selection_summary.json
  excluded_targets.jsonl
  reports/
    bucket_distribution.json
    action_distribution.json
    render_frame_reuse.json
```

## Implemented Files

- `navdp_datagen/massgen/frame_selection.py`
- `scripts/massgen/select_frame_interest_windows.py`
- `navdp_datagen/massgen/frame_selection_pilot.py`
- `scripts/massgen/prepare_frame_selection_family_pilot.py`
- `scripts/massgen/visualize_frame_selection_outputs.py`
- `scripts/massgen/render_manifest_jobs.py`
- `navdp_datagen/massgen/render_executor.py`
- `render_label_paths_telesim.py`
- `utils/telesim_path_json_outputs.py`
- `tests/test_frame_interest_selection.py`

## Validation Status

Focused local selector tests and source-side remote family validation were run
on 2026-09-08. See `docs/frame_selection_remote_validation_20260908.md` for the
paths, discovered bugs, family matrix, and remaining render validation.

The final non-rendering survey selected 26 scored POIs plus 27 mandatory anchors
across 13 ready cohorts. All 53 jobs passed structural audits; 2,728 of 3,445
retained context samples (79.2%) were forward motion. Five requested novelty
families remain waiting for source manifests.

Validated:

- selector emits 33-frame windows for selected targets;
- seeded path selection consumes complete paths and permits multiple POIs per
  path;
- assigned sub-mission endpoints and whole-mission endpoints are mandatory and
  additive to the scored POI budget;
- action ratios remain soft and center/context action distributions are reported
  separately;
- selected-window manifest rewrites render jobs without mutating source jobs;
- label-path materialization preserves repeated stationary frames;
- sparse Pathplanner waypoints are densified into contiguous renderer frames;
- all active mission families produce a family-constrained 33-frame pilot when
  a suitable local manifest exists;
- disconnected families remain explicitly `waiting_for_source_manifest` and do
  not authorize execution.

Still required:

- source scenarios for the five missing novelty families;
- Datagen asset preflight and one rendered 33-frame window per cohort.
