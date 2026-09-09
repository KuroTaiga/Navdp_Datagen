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
- 25 randomly ranked nonempty scenes per family;
- up to 50 randomly ranked complete robot source paths per selected scene;
- two scored POIs per source path, with unlimited POIs per individual path;
- mandatory endpoint/checkpoint anchors additive;
- trajectory gaps densified at 10 FPS;
- no image rendering executed.

Of 300 scene cohorts, 281 supplied all 50 paths. Nineteen cohorts contained
fewer than 50 usable generated paths, producing 14,535 paths rather than the
15,000 upper bound. The minimum cohort contained one path. No scenario
conversion or selection errors occurred.

## Aggregate Result

| Measure | Value |
| --- | ---: |
| Sampled scenes | 300 |
| Sampled scenarios | 10,834 |
| Sampled robot source paths | 14,535 |
| Source frames | 3,651,322 |
| Scored POI centers | 29,070 |
| Mandatory anchors | 21,433 |
| All selected centers | 50,503 |
| Repeated 65-frame samples | 3,282,695 |
| Unique retained source frames | 1,977,047 |
| Unique retention | 54.15% |

Repeated window execution would process 89.90% as many frame samples as full
paths. Reusing overlapping frames reduces this to 54.15% of source frames, or
60.23% of the repeated-window workload.

## Action Coverage

| Population | Move | Stop | Left | Right |
| --- | ---: | ---: | ---: | ---: |
| Scored POI centers | 60.85% | 8.71% | 15.02% | 15.43% |
| Mandatory anchors | 18.01% | 44.31% | 16.45% | 21.22% |
| All selected centers | 42.67% | 23.82% | 15.62% | 17.89% |
| Repeated 65-frame windows | 59.60% | 19.12% | 10.02% | 11.25% |
| Unique retained frames | 67.89% | 14.34% | 8.64% | 9.12% |

The scored POI pool reaches the 15% left/right objective. Stop reaches only
8.71% because several mission families have no non-anchor stop candidates;
mandatory anchors independently provide substantial stop coverage.

Compared with the previous 2,484-path survey:

| Population/action | Previous | Anchor-aware | Delta |
| --- | ---: | ---: | ---: |
| Scored POI left | 13.33% | 15.02% | +1.69 pp |
| Scored POI right | 13.04% | 15.43% | +2.39 pp |
| Repeated-window left | 3.78% | 10.02% | +6.24 pp |
| Repeated-window right | 3.87% | 11.25% | +7.38 pp |
| Unique-frame left | 3.56% | 8.64% | +5.08 pp |
| Unique-frame right | 3.66% | 9.12% | +5.46 pp |

Combined unique-frame turn coverage increased from 7.22% to 17.76%.

## Window Dilution

| Center action | Windows | Mean matching frames in 65 | At least 5 | At least 10 |
| --- | ---: | ---: | ---: | ---: |
| Left | 4,365 | 8.90 | 68.06% | 37.43% |
| Right | 4,485 | 9.07 | 68.25% | 37.99% |
| Stop | 2,532 | 32.34 | 93.48% | 88.15% |
| Move | 17,688 | 49.88 | 99.87% | 99.41% |

Turn-centered examples therefore contain about nine explicitly turning frames
on average. The remaining frames are approach/departure context, not additional
move-centered training records.

## Family Limits

`deliver_to_human`, `group_integrity`, `personal_space`, and `queue_order`
contain almost no non-anchor stop candidates. `serve_queue` has the same issue
because its meaningful stops are mandatory service/checkpoint anchors. The
availability-aware policy records these deficits and redistributes the scored
POI budget rather than relabeling weak geometric states as stops.

Aggregate unmet scored-center targets were 4 move, 396 stop, 164 left, and 184
right out of 29,070 selected POIs. Despite these scene-local deficits, aggregate
left/right coverage met the target.

## Artifacts

Local:

- `out/path_sampling_anchor_aware_20260909_50paths/report.md`
- `out/path_sampling_anchor_aware_20260909_50paths/corpus_survey.json`
- `out/path_sampling_anchor_aware_20260909_50paths/per_scene_sampling.csv`
- `out/path_sampling_anchor_aware_20260909_50paths/per_scene_counts.csv`

Remote:

- `/private_lxh/dongjk/navdata/mass_generation_runs/frame_sampling_anchor_aware_20260909_50paths/`

SHA-256:

- `corpus_survey.json`: `1a73e423fad3dc38c6ce5cc0c76e425fb79be6c79dd4d6326e4c500d657ecd8f`
- `per_scene_counts.csv`: `49d43eb7b2bfe00272135e0ea5eb97c42d189a37efee89e9caf6e37c14de64d7`
- `per_scene_sampling.csv`: `d62a241a3ec184494d26ae709871b37988d93ff672b3cfab9c478e437c65ec04`
- `report.md`: `6d7839369ef1afc9a9ae830d28ea9a2242daf77dfd2457000621b62b6927e1bb`

## Verification

- Local focused tests: 64 passed before the multiprocessing-only survey change.
- Local selector tests after multiprocessing change: 31 passed.
- Python compilation and `git diff --check`: passed.
- Remote end-to-end smoke: 24 paths across 12 families, passed.
- Remote full survey: 300 scene cohorts, 14,535 paths, zero errors.
- Remote `pytest`: unavailable because the host system Python has no `pytest`.
