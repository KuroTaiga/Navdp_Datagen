# Anchor-Aware Frame Sampling Validation - 2026-09-09

## Decision

The package contains `32 past frames + current frame` for each selected point
of interest. Future frames are not packaged, so every selected example has 33
frames. The selector still supports an explicit nonzero `future_frames` value
for experiments, but the package default is zero.

Frame selection balances scored POI centers separately from mandatory mission,
checkpoint, and trajectory-end anchors. The scored-center target is
`55% move / 15% stop / 15% left / 15% right`. Targets are availability-aware
minimums, not per-path caps. Every mandatory anchor remains additive.

Implementation commits:

- `d0c097f Balance scored frame-interest centers by action`
- `4a47cf4 Parallelize frame sampling corpus survey`
- `264e771 Package past and current sampling frames`

## Validation Scope

The corrected survey ran on `pathGen_lxh` from an isolated worktree at
`264e771`, leaving the dirty primary Datagen checkout untouched.

- formal roots: `formal_fast_v1`, `formal_slow_v1`, and
  `formal_social_slow_v1`;
- deterministic seed: `20260909`;
- all 12 mission families and every discovered nonempty scene;
- up to 50 randomly ranked complete robot source paths per scene;
- a cohort scored-POI budget equal to two times its sampled source-path count,
  with no per-path POI cap;
- mandatory endpoint/checkpoint anchors additive;
- trajectory gaps densified at 10 FPS;
- 32 historical frames, current frame, zero future frames;
- no image rendering executed.

Of 12,318 nonempty scene cohorts, 11,326 supplied all 50 paths. The remaining
992 cohorts contained fewer than 50 usable generated paths, producing 583,188
paths rather than the 615,900 upper bound. The minimum cohort contained one
path. No scenario conversion or selection errors occurred. The corrected
history-only eligibility rule filled the complete scored-POI budget.

## Aggregate Result

| Measure | Value |
| --- | ---: |
| Sampled scenes | 12,318 |
| Sampled scenarios | 434,835 |
| Sampled robot source paths | 583,188 |
| Source frames | 145,899,714 |
| Scored POI centers | 1,166,376 |
| Mandatory anchors | 880,511 |
| All selected centers | 2,046,887 |
| Repeated 33-frame samples | 67,547,271 |
| Unique retained source frames | 59,953,875 |
| Unique retention | 41.09% |

Repeated window execution would process 46.30% as many samples as all source
paths. Reusing overlapping frames reduces this to 41.09% of source frames, or
88.76% of the repeated-window workload.

## Action Coverage

| Population | Move | Stop | Left | Right |
| --- | ---: | ---: | ---: | ---: |
| Scored POI centers | 60.31% | 8.86% | 15.17% | 15.66% |
| Mandatory anchors | 17.65% | 44.74% | 16.96% | 20.65% |
| All selected centers | 41.96% | 24.29% | 15.94% | 17.81% |
| Repeated 33-frame windows | 65.59% | 13.97% | 9.91% | 10.53% |
| Unique retained frames | 66.27% | 13.68% | 9.72% | 10.33% |

The scored POI pool reaches the 15% left/right objective. Stop reaches only
8.86% because several mission families have no non-anchor stop candidates;
mandatory anchors independently provide substantial stop coverage. Training
balance must use scored-center labels or emitted per-action weights rather than
treating every retained context frame as an independent example.

## Contract Comparison

The 65-frame run used the same source roots, random seed, scene set, and path
sample. Its results are retained only for comparison.

| Measure | Former 65-frame | Current 33-frame | Change |
| --- | ---: | ---: | ---: |
| Repeated window samples | 133,047,135 | 67,547,271 | -49.23% |
| Unique retained frames | 79,815,018 | 59,953,875 | -24.88% |
| Unique source retention | 54.71% | 41.09% | -13.61 pp |
| Scored POI left | 15.11% | 15.17% | +0.06 pp |
| Scored POI right | 15.62% | 15.66% | +0.04 pp |
| Unique-frame left | 9.18% | 9.72% | +0.54 pp |
| Unique-frame right | 9.72% | 10.33% | +0.61 pp |

Removing future context saves 19,861,143 unique frame renders while preserving
the center distribution. Combined unique-frame turn coverage increases slightly
from 18.90% to 20.05%.

## Window Composition

| Center action | Windows | Mean matching frames in 33 | At least 5 | At least 10 |
| --- | ---: | ---: | ---: | ---: |
| Left | 176,956 | 5.07 | 44.98% | 11.86% |
| Right | 182,633 | 5.11 | 45.28% | 12.34% |
| Stop | 103,339 | 16.35 | 72.76% | 60.84% |
| Move | 703,448 | 24.28 | 95.94% | 88.74% |

The context now contains only the lead-up to each POI. Lower matching-action
frame counts are expected because a turn or stop may begin near the current
frame. The scored current-frame action remains the training-balance label.

## Family Limits

`deliver_to_human`, `group_integrity`, `personal_space`, and `queue_order`
contain almost no non-anchor stop candidates. Their important states are often
social or geometric events while the robot is still moving, so action balance
alone cannot represent mission criticality. Family-specific event POIs remain
the next sampling improvement.

Aggregate unmet scored-center targets were 216 move, 16,172 stop, 9,367 left,
and 9,542 right out of 1,166,376 selected POIs. These are scene-local
availability deficits; aggregate left/right coverage still meets the target.

## Artifacts

Local:

- `out/path_sampling_anchor_aware_20260909_allscenes_50paths_33frame/report.md`
- `out/path_sampling_anchor_aware_20260909_allscenes_50paths_33frame/corpus_survey.json`
- `out/path_sampling_anchor_aware_20260909_allscenes_50paths_33frame/per_scene_sampling.csv`
- `out/path_sampling_anchor_aware_20260909_allscenes_50paths_33frame/per_scene_counts.csv`

Remote:

- `/private_lxh/dongjk/navdata/mass_generation_runs/frame_sampling_anchor_aware_20260909_allscenes_50paths_33frame/`

SHA-256:

- `corpus_survey.json`: `285b3132ff0d52a9d362dadb0366c16b28af03ab7a988143f0f15211800000a3`
- `per_scene_counts.csv`: `49d43eb7b2bfe00272135e0ea5eb97c42d189a37efee89e9caf6e37c14de64d7`
- `per_scene_sampling.csv`: `efe19d06b8b7e5d8ecf9b874af317d1113e2aa3105c11c9baa356fcbcb323006`
- `report.md`: `61edc364adc7e923ad2b2569562fcadb0073e9df4515dc3edc1e7a1759d862ec`

## Verification

- Local selector/manifest/executor suite: 64 passed after the contract change.
- Python compilation and `git diff --check`: passed.
- Remote exhaustive survey: 12,318 scene cohorts, 583,188 paths, zero errors.
- Artifact audit: config is exactly `32 + 1 + 0`; all action populations sum to
  their totals; all selected centers expand to exactly 33 samples; scored POI
  budget is filled; retained frames never exceed source frames.
- No Gaussian, RGB, depth, or actor rendering was executed.
