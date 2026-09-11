# POI Policy Survey, 2026-09-11

## Decision

Use `anchor_aware_semantic_center_balance/v0.4` with an average global budget of
four scored POIs per selected source path, plus all mandatory anchors, for the
first production render pass. Keep `semantic_episode_policy=none` for that
pass. This is the best tested operating point for the current renderer, which
materializes every 33-frame window independently.

Do not use the two-POI setting as the production budget. It remains a baseline.
Do not use the full `guarantee_representative` event-aware policy for bulk
rendering yet: it is the semantic-recall upper bound, but it creates excessive
overlapping window volume. Preserve it as an opt-in metadata/export policy and
as the reference for the next selective-event experiment.

The next policy experiment should start from budget four and add guaranteed
representatives for the rare, decision-critical episode classes: traffic waits,
collision avoidance, human-interaction decisions, route deviations, instruction
turns, and target-human visibility. It should not blanket-guarantee every
instruction-section boundary or planned-execution-mismatch episode. Target at
most roughly eight scored POIs per path on average, without a per-path or
per-action cap, and measure the result on the same deterministic cohort sample.

## Survey Scope

- Inputs: `formal_fast_v1`, `formal_slow_v1`, and
  `formal_social_slow_v1` on `pathGen_lxh`.
- Seed: `20260911`.
- Population: 4,915,973 generated scenarios across 13,882 discovered
  corpus/family/scene cohorts; 12,318 cohorts contained selectable paths.
- Sample: up to 50 complete source paths in every nonempty cohort, totaling
  583,188 paths and 145,970,109 source frames.
- Path length: mean 250.30, median 301, P25 141, P75 301, P90 381, P95 381,
  P99 493, minimum 41, maximum 1,801 frames.
- Temporal contract: exactly 32 past frames plus current, no future frames;
  `target_window_index=32` and 33 frames per selected window.
- Mandatory mission, checkpoint, sub-mission, route, and stopping endpoints are
  additive. No POI-type or per-path maximum was applied.
- No Gaussian or RGB rendering was executed.

## Policy Comparison

| Policy | Scored POIs | POIs/path | Mandatory anchors | All centers/windows | Independent window frames | Window/source | Unique frames | Unique retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Budget 2 | 1,166,376 | 2.00 | 2,297,914 | 3,464,290 | 114,321,570 | 78.32% | 74,063,680 | 50.74% |
| Budget 4 | 2,332,752 | 4.00 | 2,297,914 | 4,630,666 | 152,811,978 | 104.69% | 93,366,005 | 63.96% |
| Budget 8 | 4,665,504 | 8.00 | 2,297,914 | 6,963,418 | 229,792,794 | 157.42% | 119,923,063 | 82.16% |
| Event aware | 11,219,581 | 19.24 | 2,297,914 | 13,517,495 | 446,077,335 | 305.59% | 117,713,875 | 80.64% |

Every policy covered 100% of mandatory endpoint-anchor instances. The event-aware
policy selected 2.4 times as many scored POIs as budget eight, yet retained 1.5
percentage points fewer unique frames. Its causal windows overlap heavily: it
requires 446.1 million frame occurrences despite covering 117.7 million unique
source frames. This makes it unsuitable for the current independent-window
render path, although a future render-once/cache-and-reference implementation
would change that cost calculation.

Budget four is the current volume knee. It improves unique coverage by 13.22
points over budget two while its independent-window volume remains close to one
source-corpus equivalent. Budget eight adds another 18.19 points of unique
coverage, but retains most short-path families (often more than 90%) and raises
independent volume to 1.57 corpus equivalents.

## Action Balance

| Policy/population | Move | Stop | Left | Right |
| --- | ---: | ---: | ---: | ---: |
| Source frames | 74.41% | 11.21% | 7.00% | 7.38% |
| Budget 2 POIs | 61.32% | 8.55% | 14.90% | 15.23% |
| Budget 4 POIs | 61.18% | 10.21% | 14.34% | 14.26% |
| Budget 8 POIs | 64.42% | 9.09% | 13.48% | 13.02% |
| Event-aware POIs | 46.75% | 11.51% | 20.24% | 21.50% |
| Budget 4 independent windows | 55.48% | 25.75% | 9.01% | 9.75% |
| Budget 4 unique retained frames | 65.45% | 16.50% | 8.76% | 9.29% |

The 55/15/15/15 values are availability-aware scored-center guidance, not a
required final frame mix. Budget four gives substantial and nearly symmetric
left/right center coverage while its causal windows retain forward approach
context. Stop centers are below 15%, partly because stopping-point anchors are
kept in the separate mandatory pool; stop frames still account for 25.75% of
independent budget-four windows and 16.50% of its unique retained frames.

## Semantic Episode Coverage

Mean scored-POI coverage over ten critical signals is 16.81% for budget two,
25.80% for budget four, 36.79% for budget eight, and 88.91% for event-aware.
Including mandatory anchors raises the respective means to 50.22%, 56.51%,
64.34%, and 100.00%.

Budget four scored POIs cover 29.69% of instruction-turn episodes, 24.00% of
collision-avoidance episodes, 56.73% of human-interaction decisions, 27.51% of
route deviations, 23.65% of target-human-visible episodes, and 14.87% of
traffic waits. Mandatory stopping anchors increase any-center semantic-stop
coverage to 96.80%, and endpoint coverage is 100%.

Full event-aware selection reaches 100% any-center coverage for every explicitly
guaranteed signal and 94.09% for relevant-human-visible episodes, which are not
guaranteed. Scored POIs alone cover 99.83% of instruction turns, 99.49% of room
transitions, 98.67% of semantic stops, 97.81% of route deviations, 95.81% of
target-human visibility, 92.96% of collision avoidance and execution mismatch,
and 89.10% of traffic waits. Its lower scored-only values for social-law and
human-interaction episodes reflect representatives that coincide with mandatory
anchors; the any-center metric is 100%.

These results show that fixed budgets alone cannot guarantee rare semantic
events. They also show why the next adaptive policy must be selective: blanket
coverage of 7.17 million eligible execution-mismatch episodes and 2.74 million
instruction-section boundaries dominates volume.

## Quality and Artifacts

- Survey status: success, exit code 0.
- Errors: zero cohorts with errors, zero malformed selected paths, and zero
  selected-source frames missing navigation metadata for all four experiments.
- Four compressed selection manifests passed `gzip -t` integrity checks.
- All 13 representative JSON files contain all four experiments. Across their
  573 selected chunks, every window has 33 indices, ends at its target, contains
  no future frame, and reports `target_window_index=32`.
- Visualization QA covered all four global graphs and fast, slow, and social
  BEV examples. All 13 representative GIFs are animated at 900 by 800 pixels.
- The survey output contains JSON, CSV, Markdown, four renderer-independent
  compressed JSONL selection manifests, 13 representative source-path records,
  13 BEV comparisons, 13 GIFs, and 13 instruction/metadata companions.

Remote root:
`/private_lxh/dongjk/navdata/mass_generation_runs/poi_policy_survey_20260911/optimized_final`

Local summary root:
`out/poi_policy_survey_20260911/optimized_final`

The local Markdown report contains the complete generation-population table,
per-family comparisons, navigation signal counts, and both episode-coverage
populations. The CSV and full remote JSON provide the per-scene breakdowns.

## Remaining Weaknesses

- The tested adaptive policy is a recall upper bound, not a production-volume
  policy; its frequent episode classes need tiering or deduplication.
- Availability-aware action minima cannot synthesize stop/turn candidates and
  are intentionally evaluated at POI centers rather than forced onto retained
  frames.
- Stationary stop intervals are represented by an additive stopping-point
  anchor, but dynamic-context changes inside long stops need a dedicated
  human/robot-motion ablation.
- The current renderer materializes overlapping 33-frame jobs. Rendering each
  unique source frame once and referencing it from multiple causal packages
  would materially improve the event-aware cost frontier.
- Representative BEV paths were selected deterministically per mission family;
  a later qualitative review should include the longest, most event-dense, and
  highest-overlap paths, not only one hashed representative per family.
