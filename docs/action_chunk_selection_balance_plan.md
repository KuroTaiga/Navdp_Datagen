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
render all frames `t-32..t+32`. Stationary/yield/repeated-pose frames are valid
context and must not be deduplicated away.

## Research Basis

The attached Nature D2RL paper argues for removing non-critical, uninformative
states and densifying the learner's exposure to safety-critical states. The
same principle is useful here, but the navigation task needs a deterministic
dataset selector rather than an online RL simulator edit. Fixed action ratios
such as 10/15/15/60 or 15/15/15/55 are therefore treated as starting heuristics,
not as an optimal distribution.

For NavDP-style navigation training, the selector uses two levels of balance:

1. Primary interest buckets decide which planning moments are worth rendering.
2. Secondary action ratios prevent the selected targets from becoming all
   forward motion or all stops.

Primary references:

- D2RL safety-critical state densification:
  https://www.nature.com/articles/s41586-023-05732-2
- DAgger distribution shift framing for sequential imitation learning:
  https://proceedings.mlr.press/v15/ross11a.html
- NavDP sim-to-real navigation diffusion policy and large-scale simulated data:
  https://arxiv.org/abs/2505.08712

## Default Distribution Policy

Implemented policy: `adaptive_dense_navigation/v0.1`.

Primary interest buckets:

| Bucket | Meaning |
| --- | --- |
| `critical_margin` | low clearance, near-collision margin, safety event, or near actor/robot |
| `human_interaction` | target/bystander human is close enough to affect the path |
| `route_decision` | turn apex, stop/yield transition, release/deadline/event proximity |
| `representative_motion` | clean motion needed for coverage after the informative states are sampled |

Default bucket priors favor critical and interaction states, then route
decisions, while bounding every bucket so representative motion remains present.
The actual target ratios are derived from the available candidate distribution
per run. Operators can override them with `--bucket-ratio`.

Secondary action target ratios default to:

| Action | Code | Default Share |
| --- | ---: | ---: |
| stop | `0` | 20% |
| forward / move | `1` | 40% |
| turn left | `2` | 20% |
| turn right | `3` | 20% |

These ratios are configurable with `--action-ratio`. They are deliberately less
forward-heavy than the older 45-60% move targets because forward frames also
enter the rendered set as context around stops, turns, and interactions.

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

Every camera trajectory sample that can support the requested window becomes a
candidate. The default edge policy is `reject`, so a 65-frame run only selects
targets with 32 valid past and 32 valid future frames. `clamp` is available for
explicit padded-edge experiments.

Each candidate stores:

- source manifest index/path and source render job id;
- source frame index/id, timestamp, pose, yaw, and coarse action;
- all 65 source frame indices required for rendering;
- interest bucket scores and selection reasons;
- split and target role metadata.

Score components include actor/robot proximity, event proximity, stop/turn
transitions, turn magnitude, and representative clean-motion coverage.

## Selection Algorithm

1. Load MassGen render manifests.
2. Build per-frame candidates from every render job trajectory.
3. Reject candidates that cannot satisfy the requested window under the edge
   policy.
4. Derive adaptive interest-bucket ratios from available candidates unless the
   run provides explicit bucket ratios.
5. Convert bucket ratios to integer target counts.
6. Select the best candidates per bucket using interest score, secondary action
   deficits, deterministic seed jitter, per-job caps, and minimum target spacing.
7. Fill any remaining target count from the global best candidates.
8. Emit a selection manifest and optional selected-window render manifest.

The target frame action is used for balance; surrounding context frames do not
count toward the action histogram.

## Selection Manifest

`scripts/massgen/select_frame_interest_windows.py` writes
`navdp_frame_interest_selection/v0.1` manifests:

```json
{
  "schema_version": "navdp_frame_interest_selection/v0.1",
  "config": {
    "past_frames": 32,
    "future_frames": 32,
    "window_frame_count": 65,
    "distribution_policy": "adaptive_dense_navigation/v0.1"
  },
  "distribution": {
    "target_bucket_ratios": {
      "critical_margin": 0.4,
      "human_interaction": 0.25,
      "representative_motion": 0.1,
      "route_decision": 0.25
    },
    "target_action_ratios": {
      "move": 0.4,
      "stop": 0.2,
      "turn_left": 0.2,
      "turn_right": 0.2
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
  --target-count 1000 \
  --output-selection-json out/frame_selection/selection_manifest.json
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

Current implementation renders one selected-window job per selected target. Each
window job contains a 65-point camera trajectory, rewrites renderer-local frame
ids to `0..64`, and preserves original source frame indices in metadata.

The renderer path now respects `camera.preserve_frame_samples` and
`metadata.preserve_frame_samples`, so repeated stationary poses are kept in the
label path and are not removed by path deduplication or distance resampling.
Executor planning also blocks `--minimal-frames` values that would truncate a
selected frame-interest window.

The selection summary reports both:

- `requested_window_frame_renders`: selected targets multiplied by 65;
- `unique_source_render_frame_count`: de-duplicated source frames needed across
  overlapping windows.

A future sparse-frame renderer can render each unique source frame once and map
it back to every chunk. The current selected-window implementation is the
practical next step because it prevents full-path rendering while preserving the
65-frame model contract.

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
- `scripts/massgen/render_manifest_jobs.py`
- `navdp_datagen/massgen/render_executor.py`
- `render_label_paths_telesim.py`
- `utils/telesim_path_json_outputs.py`
- `tests/test_frame_interest_selection.py`

## Validation Status

Tests were added but not executed in this pass because testing is currently not
available per operator instruction.

Planned validation when testing resumes:

- selector emits 65-frame windows for selected targets;
- selected-window manifest rewrites render jobs without mutating source jobs;
- label-path materialization preserves repeated stationary frames;
- action and bucket deficits are reported when quotas are impossible;
- edge-window `reject` and `clamp` policies are covered.
