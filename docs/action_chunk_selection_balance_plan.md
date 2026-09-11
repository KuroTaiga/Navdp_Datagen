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
  "future_context_frame_count": 0,
  "render_window_frame_count": 33
}
```

If frame `t` is selected for training or testing, the renderer must preserve and
render all frames `t-32..t`. At a required path endpoint, the nearest edge
sample is repeated when needed so the endpoint remains the final sample. Stationary, yield,
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
2. Secondary action ratios provide selection priority and availability-aware
   scored-center minimums. They impose no per-action maximum and do not affect
   the additive mandatory endpoints.

Primary references:

- D2RL safety-critical state densification:
  https://www.nature.com/articles/s41586-023-05732-2
- DAgger distribution shift framing for sequential imitation learning:
  https://proceedings.mlr.press/v15/ross11a.html
- NavDP sim-to-real navigation diffusion policy and large-scale simulated data:
  https://arxiv.org/abs/2505.08712

## Default Distribution Policy

Implemented policy: `anchor_aware_semantic_center_balance/v0.6`.

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
soft selection priority and availability-aware scored-center minimums. They do
not impose per-path or per-action maximums. The survey reports action
distributions at selected centers and across the complete retained 33-frame
windows. This avoids mistaking a critical-state-heavy center distribution for a
dataset with no forward motion.

## Inputs

The implemented selector reads one or more MassGen render manifests generated
from Pathplanner scenarios. It uses:

- `jobs[*].camera.trajectory` for frame, time, pose, yaw, and motion state;
- mission release/deadline times and event logs for route-decision salience;
- manifest human trajectories and peer-robot pose tracks for proximity scores;
- job family and actor ids for chunk metadata and downstream filtering;
- deterministic Pathplanner navigation supervision on every trajectory sample,
  including instruction sections, room membership, planned/executed actions,
  route deviation, decision provenance, and target-human visibility.

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
transitions, turn magnitude, and representative clean-motion coverage. The v0.6
policy also derives the following auditable navigation signals from Pathplanner
metadata:

- instruction turns, semantic stops, and instruction-section boundaries;
- room transitions;
- collision avoidance and primary or secondary L1-L4 social-law decisions;
- human-interaction decisions, traffic waits, and visible relevant/target humans;
- mission endpoint stops;
- route deviation and planned/executed action mismatch.

These signals boost the applicable interest-bucket score and are emitted on each
selected chunk. They are not separate quotas. This keeps the configured action
distribution effective while choosing semantically stronger points within each
action and interest stratum. Reports include both signal-bearing frame counts
and contiguous per-path signal episodes, plus the number of distinct episodes
covered by at least one scored POI. Episode coverage prevents a long stop or
avoidance interval from being misread as many independent events.

The production default uses `guarantee_representative` and chooses one scored
apex per important contiguous episode. A long episode receives approach or
completion centers only
when its action, instruction, room, decision reason, or visible-human context
changes. With the default zero target count there is no artificial scored-POI
floor: the event representatives define the count. A positive target count is
an optional global minimum, never a per-path or per-action cap.
`semantic_episode_policy=none` remains available only for fixed-budget
comparisons and ablations.

## Selection Algorithm

1. Load MassGen render manifests and apply mission-family filtering.
2. Deterministically random-rank complete source paths using the run seed, then
   retain all paths or the requested `--max-source-paths` subset.
3. Densify each selected path and score every eligible frame on the full path.
4. Mark assigned sub-mission completion/checkpoint frames, each trajectory end,
   and the end of each contiguous physical stop episode as mandatory anchors.
   A long stationary interval contributes one stopping-point anchor, not one
   anchor per stopped frame.
5. Derive sparse representatives of every guaranteed semantic episode.
6. Treat a positive target count as an optional global minimum; the default zero
   count adds no artificial floor in event-aware mode.
7. Add mandatory anchors first. They are additive and do not consume the scored
   POI count, per-path POI cap, or action-deficit counts.
8. Select all episode representatives, then fill only a positive unmet minimum
   using bucket score, soft action deficit, deterministic seed jitter, and
   spacing. Multiple POIs may come from the same path;
   `--max-targets-per-job 0` leaves that unlimited.
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
    "distribution_policy": "anchor_aware_semantic_center_balance/v0.6"
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
      "target_window_index": 32,
      "target_action_name": "turn_left",
      "target_bucket": "route_decision",
      "navigation_signals": [
        "instruction_turn",
        "instruction_section_boundary",
        "room_transition"
      ],
      "window": {
        "source_frame_indices": [88, 89, 90],
        "render_frame_count": 33
      },
      "render_contract": {
        "must_render_all_window_frames": true,
        "preserve_stationary_frames": true,
        "target_renderer_frame_index": 32
      }
    }
  ]
}
```

The sample shortens `source_frame_indices` for readability. Real manifests write
all 33 indices. `target_window_index` identifies the current frame explicitly;
with the default past-only contract it is the final window frame.

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

The visualization output includes path/selected-window BEV GIFs, action and
interest-bucket distributions, and `navigation_signal_coverage.png` for the
metadata-grounded signal populations and scored-POI retention rates.

Run the all-cohort fixed/adaptive policy comparison without rendering:

```bash
python3 scripts/analysis/survey_poi_policy_matrix.py \
  --formal-fast-root /path/to/formal_fast_v1 \
  --formal-slow-root /path/to/formal_slow_v1 \
  --formal-social-root /path/to/formal_social_slow_v1 \
  --output-dir out/poi_policy_matrix \
  --paths-per-cohort 50 \
  --scenes-per-family 0 \
  --seed 20260911 \
  --workers 119
```

The matrix compares average global scored-POI budgets of 2, 4, and 8 per
sampled path with the adaptive episode-representative policy. It discovers
cohorts once, reads only deterministically selected scenarios, emits compressed
renderer-independent selection manifests, and reports all-source, center,
independent-window, and unique-retained populations separately.

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
- `scripts/analysis/survey_poi_policy_matrix.py`
- `scripts/analysis/visualize_poi_policy_matrix.py`
- `scripts/analysis/run_poi_policy_matrix_remote.sh`
- `scripts/analysis/watch_remote_poi_policy_survey.sh`
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
