# Frame-Selection Remote Validation - 2026-09-08

> Historical contract: this validation exercised `32 past + current + 32
> future`. The package default changed on 2026-09-09 to `32 past + current`
> (33 frames), so these counts must not be used for current render sizing.

## Scope

The frame-interest sampler was exercised through the `pathGen_lxh` SSH target
against representative Pathplanner scenarios from:

- `/private_lxh/dongjk/navdata/mass_generation_runs/formal_fast_v1`;
- `/private_lxh/dongjk/navdata/mass_generation_runs/formal_slow_v1`;
- `/private_lxh/dongjk/navdata/mass_generation_runs/formal_social_slow_v1`.

The dirty remote Datagen and Pathplanner worktrees were not modified. Test code
was copied to an isolated overlay under:

```text
/private_lxh/dongjk/navdata/mass_generation_runs/frame_sampling_validation_20260908
```

This validation covered scenario conversion, frame selection, selected render
manifest construction, and structural audits. It did not execute Gaussian
rendering.

## Bugs Found And Fixed

1. Pathplanner trajectories are sparse time-stamped waypoints. A 30-second path
   can contain only 64 waypoint objects while its renderer frame ids span
   `0..300`. Counting waypoint objects made valid 65-frame windows ineligible.
   The selector now interpolates position, time, and wrapped yaw across integer
   frame-id gaps before selection and records interpolation provenance.
2. Mixed `mission_stream` manifests also advertise embedded mission families.
   They could displace dedicated family scenarios. The pilot now prefers
   dedicated single-family manifests and falls back to mixed manifests only
   when no dedicated source exists.
3. Social-navigation subfamilies were collapsed into
   `navigate_with_social_constraints`. The pilot now provides separate cohorts
   for personal space, pedestrian yield, group integrity, and queue order based
   on `L1` through `L4` law ids.
4. One pre-existing test treated renderer world points as arrays instead of
   `{x,y,z}` objects. The assertion was corrected.
5. The first visualization showed only selected-center actions, which made the
   selection appear to remove forward motion. Summaries and plots now separate
   all selected centers, scored POI centers, repeated 65-frame context samples,
   and unique retained source frames.
6. The ordinary `reject` edge policy excluded final checkpoints because they do
   not have 32 future frames. Assigned sub-mission completions and trajectory
   ends are now mandatory anchors with an independent `clamp` endpoint policy.
7. Mandatory anchors initially consumed the requested POI budget. They are now
   additive, so `target_count=2` means two scored POIs plus every required
   checkpoint/end anchor.
8. Candidate selection previously operated directly over all path frames. The
   selector now seed-samples complete source paths first, scores each selected
   path in full, and permits multiple POIs from one path.

## Results

Local focused suite:

```text
30 passed in 0.35s
```

Broader selector/manifest/executor suite:

```text
65 passed in 0.35s
```

Remote matrix:

```text
18 requested cohorts
13 ready cohorts
5 waiting_for_source_manifest
26 scored points of interest
27 mandatory checkpoint/end anchors
53 selected target jobs
3,445 requested window-frame renders
2,393 unique source frames
0 structural audit errors
```

Every ready selected job contained exactly 65 source samples, placed its target
at index 32, used only source-frame steps of zero or one, and had stationary
sample preservation enabled. Zero steps are expected when a mandatory endpoint
uses edge padding. Every generated command resolved to
`/team/telenav/code/Navdp_Datagen/scripts/massgen/render_manifest_jobs.py`.

The survey selected at most two complete source paths per cohort. All 13 ready
cohorts selected multiple target centers from at least one chosen path. Action
ratios were not used as hard quotas:

| Population | Move | Stop | Turn left | Turn right |
| --- | ---: | ---: | ---: | ---: |
| Selected centers (53) | 36 | 15 | 2 | 0 |
| Scored POI centers (26) | 26 | 0 | 0 | 0 |
| All retained window samples (3,445) | 2,728 | 618 | 28 | 71 |

Forward motion therefore makes up 79.2% of retained model input samples even
though mandatory endpoints intentionally add stop/checkpoint centers. This is
why center-role composition and full-window action composition must be assessed
separately.

Ready top-level families:

- `human_guided_uncertain_region`
- `serve_queue`
- `dense_dynamic_humans`
- `dense_dynamic_combined`
- `dense_dynamic_avoidance`
- `dense_multi_robot`
- `mission_stream`
- `deliver_to_human`
- `navigate_with_social_constraints`

Ready social-law cohorts:

- `navigate_with_social_constraints:personal_space`
- `navigate_with_social_constraints:pedestrian_yield`
- `navigate_with_social_constraints:group_integrity`
- `navigate_with_social_constraints:queue_order`

No source directory was present in the supplied formal roots for:

- `interruption_recovery`
- `multi_robot_handoff`
- `escort_and_rendezvous`
- `implicit_need_fulfillment`
- `conflict_resolution`

The final remote pilot plan is:

```text
/private_lxh/dongjk/navdata/mass_generation_runs/frame_sampling_validation_20260908/results/pilot_whole_path_poi_v3/pilot_plan.json
```

Local non-rendered visual outputs are under:

```text
out/frame_sampling_visualization_whole_path_poi_v3
```

The combined GIF and per-family BEV overlays distinguish complete selected
paths, 65-frame windows, scored POIs, mandatory endpoints, unselected eligible
paths, and human tracks. The distribution plot compares candidate buckets and
actions against selected centers and retained context frames.

## Remaining Validation

1. Generate or connect Pathplanner source scenarios for the five missing
   novelty families and rerun their family entries.
2. Connect Datagen scene and actor assets, run every generated `plan_argv`, and
   resolve preflight blockers.
3. Explicitly authorize and execute one 65-frame render job per cohort.
4. Audit video/camera metadata, actor metadata, source-frame identity, and
   output frame count before increasing the per-family target budget.
