# Datagen Action Chunk Selection And Balance Plan

Status: design plan
Repo: `Navdp_Datagen`
Scope: select target frames, expand required history frames, and rebalance action
labels before expensive rendering or dataset packaging.

## Purpose

The pathplanner-side selector should decide which generated mission/path
candidates are worth sending downstream. This Datagen-side selector starts after
that step. It decides which target frames are useful training examples, which
history frames those targets require, and which unique frames must be rendered.

The immediate action-balance target is:

| Action | Code | Target Share |
| --- | ---: | ---: |
| stop | `0` | 15% |
| forward / move | `1` | 45% remainder |
| turn left | `2` | 20% |
| turn right | `3` | 20% |

The selector should optimize the selected target frames for this distribution
while preserving mission-interesting frames and avoiding bad histories.

## Local Evidence And Contract

Current local code confirms the action payload convention:

- `scripts/actions/export_frame_actions.py` maps `stop=0`, `move=1`,
  `turn left=2`, and `turn right=3`.
- `scripts/actions/export_frame_actions.py` defaults `--max-next` to `8`.
- `utils/telesim_path_json_outputs.py` exports `curr_action` and
  `next_actions`, and also has a configurable reverse-history payload with
  `prev_actions` and `prev_frames`.
- `scripts/datasets/generate_vlnpe_dataset.py` consumes
  `curr_action + next_actions[:future_actions]`.

The user-requested production assumption is:

```json
{
  "history_frame_count": 32,
  "actions_per_chunk": 8,
  "frames_per_action": 4
}
```

That should be treated as the new selector contract, but it still needs to be
checked against the model-side configuration. The current local
`generate_vlnpe_dataset.py` default is `--history-frames 16` and
`--future-actions 4`, so the selector must not silently rely on old defaults.
Every manifest should store the history/action/chunk dimensions used for that
run.

## Definitions

- Source frame: a frame index in the original planned/rendered path timeline.
- Target frame: a frame selected as a supervised training example and counted
  toward action balance.
- History frame: a previous frame required as context for a target frame.
- Render frame: any frame that must be rendered. This is the union of all target
  frames and all history frames.
- Chunk: one target frame plus its history and action horizon. This is a dataset
  selection unit, not necessarily a renderer video chunk.

For the 32-history contract, target frame `t` normally requires frames
`t-31..t`. If the model contract means 32 previous frames plus current, then the
config should be changed to `history_frame_count=33`; the manifest structure is
the same.

## Non Goals

- Do not decide whether a path candidate is planner-quality good enough. That is
  owned by the pathplanner candidate filter.
- Do not change planned trajectories.
- Do not rebalance by deleting stops caused by real yielding or waiting.
- Do not allow recovery jumps in context histories for normal training data.
- Do not require every selected target to show humans; that depends on mission
  family and pathplanner salience metadata.

## Input Artifacts

The selector should read:

- ranked path candidate manifest from the pathplanner repo;
- mission JSON and actor trajectories;
- action payloads with `curr_action` and `next_actions`;
- pathplanner interaction windows and `avoid_history_ranges`;
- optional pre-render visibility estimates such as relevant humans, objects, or
  peer robots near the camera path;
- renderer capability/config metadata, including whether sparse frame rendering
  is supported.

The pathplanner manifest provides family-aware salience. Datagen uses that
signal to choose frames, but action balancing is done here.

## Frame Feature Table

Normalize every input path into a frame table before selection:

```json
{
  "scene_id": "0011_859081",
  "label_id": "example_CHINGMU_rescaled_3_0011_859081_dense_dynamic_humans_v9944",
  "frame": 144,
  "timestamp_s": 9.0,
  "curr_action": 2,
  "next_actions": [1, 1, 2, 1, 0, 1, 1, 1],
  "robot_pose_world": [12.3, 4.5, 1.1],
  "in_interaction_window": true,
  "interaction_window_ids": ["robot_001_human_normal_01_crossing_0001"],
  "family_salience": {
    "score": 0.87,
    "relevant_human_count_est": 1,
    "relevant_robot_count_est": 0,
    "relevant_object_count_est": 0,
    "min_relevant_actor_distance_m": 1.2
  },
  "recovery": {
    "is_jump_frame": false,
    "history_crosses_jump": false,
    "source_event_ids": []
  }
}
```

This table lets selection stay independent of renderer internals.

## Selection Algorithm

1. Load the ranked path candidate manifest and action payloads.
2. Build the normalized frame feature table for each candidate.
3. Mark timeline discontinuities from `avoid_history_ranges`, teleport/recovery
   metadata, and direct pose-jump detection.
4. Compute `history_frames` for each possible target frame from the configured
   `history_frame_count`.
5. Reject target frames whose histories cross recovery jumps or unavailable
   timeline gaps.
6. Score eligible targets using action rarity, mission salience, interaction
   windows, diversity, and render efficiency.
7. Select targets with a quota-aware sampler until the requested action ratios
   are reached per run, per family, or per configured batch.
8. Expand selected targets into unique render frames.
9. Write a selection manifest and a renderer frame manifest.

The action label used for balancing should initially be `curr_action`. If later
training wants to balance by the eight-action sequence pattern, add a separate
`sequence_class` field instead of overloading `curr_action`.

## Scoring

A target frame score should combine:

- action deficit weight: how far the current selected set is from the target
  share for that action;
- family salience: inherited from pathplanner interaction-window scoring;
- temporal importance: start/end of stops, turn apexes, closest-approach frames,
  delivery/contact frames, queue/service transitions;
- visibility estimate: relevant human/object/robot likely in view when relevant
  to the mission family;
- diversity: scene, family, actor identity, route shape, and interaction type;
- render efficiency: prefer targets whose histories overlap already selected
  render frames, after preserving quality constraints;
- recovery penalty: reject or heavily penalize frames near jumps.

Forward frames should be sampled sparsely unless they are part of a meaningful
approach, recovery-free history, or interaction context.

## Downsampling Long Forward Runs

Long forward-only segments with little visible interaction are the main source of
action imbalance and wasted rendering. Segment the frame table into runs with the
same coarse behavior:

- stop run;
- left-turn run;
- right-turn run;
- forward run;
- mixed interaction run.

For forward runs:

- keep endpoints near stops, turns, goals, actor encounters, and room
  transitions;
- keep a small periodic sample for route continuity;
- lower sampling probability when no relevant human/object/robot is near or
  visible;
- increase sampling probability if the run is needed as history for a selected
  stop or turn target;
- avoid selecting target frames whose only value is repeated empty straight
  travel.

This downsampling should happen at target-frame selection time. History expansion
may still cause some forward frames to be rendered, but those history-only frames
do not count toward action balance.

## Recovery Jumps And Bad Histories

Planner recovery can create frame jumps for unavoidable collision or deadlock
resolution. Those are useful debugging signals, but normal model histories should
not include them.

Detection sources:

- pathplanner `avoid_history_ranges`;
- `corner_case_recovery` events;
- per-actor `teleport_count` or `teleport_recovery` state;
- robot pose displacement above a configured per-frame threshold;
- timestamp/frame gaps beyond expected stride.

Policy for normal training:

- reject target frames when `history_crosses_jump=true`;
- reject target frames that are themselves jump/recovery frames;
- allow later opt-in debug splits for recovery-specific training, but mark them
  with `split=debug_recovery`;
- never hide the jump by relabeling frames as smooth history.

If a target is early in a clean segment and does not have enough prior frames,
either reject it or use explicit padding only when the model contract accepts
padding. The manifest must record which policy was used.

## Selection Manifest

Write one manifest per selection run:

```json
{
  "schema_version": "navdp_action_chunk_selection.v0.1",
  "run_id": "20260814_chingmu3_action_balance_v1",
  "config": {
    "history_frame_count": 32,
    "actions_per_chunk": 8,
    "frames_per_action": 4,
    "target_action_ratios": {
      "stop": 0.15,
      "move": 0.45,
      "turn_left": 0.20,
      "turn_right": 0.20
    },
    "history_padding_policy": "reject_if_short",
    "recovery_history_policy": "reject_crossing_history"
  },
  "source_runs": [
    {
      "path_candidate_manifest": ".../path_candidate_quality/quality.jsonl",
      "renderer_profile": "5880_or_h100_profile_name"
    }
  ],
  "chunks": [
    {
      "chunk_id": "0011_859081_dense_dynamic_humans_v9944_t00120",
      "scene_id": "0011_859081",
      "label_id": "example_CHINGMU_rescaled_3_0011_859081_dense_dynamic_humans_v9944",
      "mission_family": "dense_dynamic_humans",
      "target_frame": 120,
      "target_action": 2,
      "target_action_name": "turn_left",
      "target_role": "balanced_label",
      "history_frames": [113, 114, 115, 116, 117, 118, 119, 120],
      "history_frame_count": 32,
      "future_actions": [1, 1, 2, 1, 0, 1, 1, 1],
      "render_frames": [113, 114, 115, 116, 117, 118, 119, 120],
      "history_valid": true,
      "history_invalid_reason": null,
      "recovery_event_overlap": false,
      "interaction_window_ids": ["robot_001_human_normal_01_crossing_0001"],
      "salience": {
        "score": 0.87,
        "relevant_human_count_est": 1,
        "min_relevant_actor_distance_m": 1.2
      },
      "source_paths": {
        "mission_json": "...",
        "action_json": "...",
        "bev_png": "...",
        "bev_gif": "..."
      }
    }
  ],
  "render_frame_index": {
    "0011_859081/example_CHINGMU_rescaled_3_0011_859081_dense_dynamic_humans_v9944": {
      "target_frames": [120],
      "history_only_frames": [113, 114, 115, 116, 117, 118, 119],
      "all_render_frames": [113, 114, 115, 116, 117, 118, 119, 120],
      "frame_roles": {
        "113": ["history"],
        "120": ["target", "history"]
      }
    }
  },
  "balance_summary": {
    "target_count": 1000,
    "action_counts": {
      "stop": 150,
      "move": 450,
      "turn_left": 200,
      "turn_right": 200
    },
    "unique_render_frame_count": 11840,
    "render_frames_per_target": 11.84
  }
}
```

The sample shortens `history_frames` for readability. Real manifests must write
the complete configured history list.

## Renderer Integration

Preferred path:

- write a sparse `render_frame_manifest.jsonl` grouped by scene and label;
- update render execution to accept explicit frame ids;
- preserve original frame ids, timestamps, camera poses, actor poses, and action
  metadata;
- render each unique frame once, then map it back to all chunks that need it;
- write debug videos only as review artifacts, not as the source of truth for
  selected frame identity.

Fallback path:

- render the full path;
- extract target/history frames from the full output;
- record the inefficiency in the run report.

The H100 pipeline should prefer sparse rendering plus CPU-side video/image
encoding, because the H100 platform has strong GPU rendering capacity and large
CPU/RAM headroom but no NVENC/RT encoder path.

## Output Layout

Suggested local layout:

```text
out/action_chunk_selection/<run_id>/
  selection_manifest.json
  render_frame_manifest.jsonl
  action_balance_report.json
  excluded_targets.jsonl
  qa/
    action_distribution.png
    render_frame_reuse.png
    selected_windows_by_family.json
```

When rendering follows selection:

```text
out/action_chunk_selection/<run_id>/renders/
  <scene_id>/<label_id>/
    frames/
    chunks/
    debug_video.mp4
    render_log.txt
```

## Baseline And Comparison

Before optimization, record:

- action counts from the existing full path or simple stride selection;
- number of full frames rendered;
- number of target samples produced;
- render time and log output;
- post-render visual QA summary.

After optimization, record:

- selected target action ratios;
- unique render frame count and render-frames-per-target;
- discarded forward-frame percentage;
- rejected target count by reason;
- recovery-history rejection count;
- target frame coverage by mission family and interaction window;
- render time and log output using the same scene/family set.

The baseline and optimized run reports should be diffable JSON so we can compare
quality and speed without manually inspecting every video.

## Implementation Plan

1. Add a selector config dataclass with action ratios, history size, chunk size,
   padding policy, and recovery policy.
2. Add a frame-table builder for mission/action/pathplanner candidate artifacts.
3. Add recovery-jump detection and history validity checks.
4. Add target scoring and quota-aware selection.
5. Add render-frame deduplication and manifest writing.
6. Add a CLI, for example `scripts/massgen/select_action_chunks.py`.
7. Add renderer support for sparse frame manifests, or implement the full-render
   fallback first if sparse rendering needs more surgery.
8. Add baseline/optimized comparison reports.

## Tests

- Unit test that action quotas converge to 15/45/20/20 when enough candidates
  exist.
- Unit test that impossible quotas degrade with explicit deficit reporting.
- Unit test that target frames and history-only frames are stored separately.
- Unit test that render-frame deduplication preserves all chunk mappings.
- Unit test that histories crossing recovery jumps are rejected.
- Unit test that long empty forward runs are downsampled but stop/turn windows
  are preserved.
- Fixture test using a CHINGMU planned mission with known stop/turn/human
  interaction frames.
- Golden JSON test for `selection_manifest.json` schema stability.

## Open Checks

- Confirm the model-side contract for `history_frame_count=32`, whether current
  frame is included, and whether VAE/history dimensions match this exactly.
- Confirm whether balancing should be by `curr_action` only or by the 8-action
  sequence pattern.
- Confirm whether early clean-segment targets may use explicit padded history or
  should always be rejected.
- Confirm if artificial terminal stop padding should be excluded from stop quota;
  default should be exclude.
