# Anchor-Aware Frame Sampling Validation - 2026-09-09

## Decision

Frame selection now balances scored frame-of-interest centers separately from
mandatory mission, checkpoint, and trajectory-end anchors. The scored-center
target is `55% move / 15% stop / 15% left / 15% right`. Targets are minimums,
not per-path caps, and unavailable action capacity is redistributed. Every
mandatory anchor remains selected and every selected center retains its full
`32 past + current + 32 future` context.

The selection manifest also emits:

- mandatory-anchor action counts as a separate population;
- scored-center action deficits when a requested action is unavailable;
- per-action training weights for center-balanced loading;
- action composition conditioned on the scored center action;
- a unique source-frame index for reuse across overlapping windows.

Implementation commits:

- `d0c097f Balance scored frame-interest centers by action`
- `4a47cf4 Parallelize frame sampling corpus survey`

## Validation Scope

The survey ran on `pathGen_lxh` from an isolated worktree at `4a47cf4`, leaving
the dirty primary Datagen checkout untouched.

- formal roots: `formal_fast_v1`, `formal_slow_v1`, and
  `formal_social_slow_v1`;
- deterministic seed: `20260909`;
- 12 mission-family cohorts;
- every nonempty scene discovered for each family;
- up to 50 randomly ranked complete robot source paths per selected scene;
- a cohort scored-POI budget of two times its sampled source-path count, with
  no per-path POI cap;
- mandatory endpoint/checkpoint anchors additive;
- trajectory gaps densified at 10 FPS;
- no image rendering executed.

Of 12,318 nonempty scene cohorts, 11,326 supplied all 50 paths. The remaining
992 cohorts contained fewer than 50 usable generated paths, producing 583,188
paths rather than the 615,900 upper bound. The minimum cohort contained one
path. No scenario conversion or selection errors occurred. Two cohorts lacked
enough valid candidates to fill the nominal scored-POI budget: one was short by
four and one was short by four, for an aggregate deficit of eight POIs.

## Aggregate Result

| Measure | Value |
| --- | ---: |
| Sampled scenes | 12,318 |
| Sampled scenarios | 434,835 |
| Sampled robot source paths | 583,188 |
| Source frames | 145,899,714 |
| Scored POI centers | 1,166,368 |
| Mandatory anchors | 880,511 |
| All selected centers | 2,046,879 |
| Repeated 65-frame samples | 133,047,135 |
| Unique retained source frames | 79,815,018 |
| Unique retention | 54.71% |

Repeated window execution would process 91.19% as many frame samples as full
paths. Reusing overlapping frames reduces this to 54.71% of source frames, or
59.99% of the repeated-window workload.

## Action Coverage

| Population | Move | Stop | Left | Right |
| --- | ---: | ---: | ---: | ---: |
| Scored POI centers | 60.44% | 8.83% | 15.11% | 15.62% |
| Mandatory anchors | 17.65% | 44.74% | 16.96% | 20.65% |
| All selected centers | 42.03% | 24.28% | 15.91% | 17.78% |
| Repeated 65-frame windows | 58.68% | 19.26% | 10.45% | 11.61% |
| Unique retained frames | 66.82% | 14.28% | 9.18% | 9.72% |

The scored POI pool reaches the 15% left/right objective. Stop reaches only
8.83% because several mission families have no non-anchor stop candidates;
mandatory anchors independently provide substantial stop coverage.

Compared with the previous 2,484-path survey:

| Population/action | Previous | Anchor-aware | Delta |
| --- | ---: | ---: | ---: |
| Scored POI left | 13.33% | 15.11% | +1.78 pp |
| Scored POI right | 13.04% | 15.62% | +2.58 pp |
| Repeated-window left | 3.78% | 10.45% | +6.67 pp |
| Repeated-window right | 3.87% | 11.61% | +7.74 pp |
| Unique-frame left | 3.56% | 9.18% | +5.62 pp |
| Unique-frame right | 3.66% | 9.72% | +6.06 pp |

Combined unique-frame turn coverage increased from 7.22% to 18.90%.

## Window Dilution

| Center action | Windows | Mean matching frames in 65 | At least 5 | At least 10 |
| --- | ---: | ---: | ---: | ---: |
| Left | 176,262 | 9.19 | 69.55% | 39.70% |
| Right | 182,193 | 9.26 | 69.08% | 39.48% |
| Stop | 102,995 | 31.68 | 93.33% | 87.50% |
| Move | 704,918 | 49.18 | 99.84% | 99.34% |

Turn-centered examples therefore contain about nine explicitly turning frames
on average. The remaining frames are approach/departure context, not additional
move-centered training records.

## Family Limits

`deliver_to_human`, `group_integrity`, `personal_space`, and `queue_order`
contain almost no non-anchor stop candidates. `serve_queue` has the same issue
because its meaningful stops are mandatory service/checkpoint anchors. The
availability-aware policy records these deficits and redistributes the scored
POI budget rather than relabeling weak geometric states as stops.

Aggregate unmet scored-center targets were 276 move, 16,443 stop, 7,183 left,
and 7,189 right out of 1,166,368 selected POIs. Despite these scene-local
deficits, aggregate left/right coverage met the target.

## Artifacts

Local:

- `out/path_sampling_anchor_aware_20260909_allscenes_50paths/report.md`
- `out/path_sampling_anchor_aware_20260909_allscenes_50paths/corpus_survey.json`
- `out/path_sampling_anchor_aware_20260909_allscenes_50paths/per_scene_sampling.csv`
- `out/path_sampling_anchor_aware_20260909_allscenes_50paths/per_scene_counts.csv`

Remote:

- `/private_lxh/dongjk/navdata/mass_generation_runs/frame_sampling_anchor_aware_20260909_allscenes_50paths/`

SHA-256:

- `corpus_survey.json`: `7c70b530c74767bb185092ba429c24c4a80458b25bdb92b9aff38f73ee5ce87e`
- `per_scene_counts.csv`: `49d43eb7b2bfe00272135e0ea5eb97c42d189a37efee89e9caf6e37c14de64d7`
- `per_scene_sampling.csv`: `ad229beba1955d7aceee7f2a87be172e593b685335e41b26b45b3717aa64a8fb`
- `report.md`: `20abeee5d476457ad796a94638928863c60e304c11e26d65126322894dd7f15a`

## Verification

- Local focused tests: 64 passed before the multiprocessing-only survey change.
- Local selector tests after multiprocessing change: 31 passed.
- Python compilation and `git diff --check`: passed.
- Remote end-to-end smoke: 24 paths across 12 families, passed.
- Remote stratified survey: 300 scene cohorts, 14,535 paths, zero errors.
- Remote exhaustive survey: 12,318 scene cohorts, 583,188 paths, zero errors.
- Artifact audit: all action populations sum to their reported totals, all
  selected centers expand to exactly 65 repeated frames, and retained frames
  never exceed source frames. The scored-POI total is eight below its nominal
  `2 * source paths` cohort budget due to unavailable valid candidates in two
  scenes.
- Remote `pytest`: unavailable because the host system Python has no `pytest`.
