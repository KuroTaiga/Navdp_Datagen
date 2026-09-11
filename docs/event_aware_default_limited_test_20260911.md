# Event-Aware Default Limited Test, 2026-09-11

## Result

`anchor_aware_semantic_center_balance/v0.6` passed the limited H100 validation
and is now the canonical selector default. The default
`semantic_episode_policy` is `guarantee_representative`; fixed-budget selection
remains available explicitly as `none` for comparisons and ablations.
The default target count is zero, so no artificial POI floor is added; detected
episode representatives define the scored count.

The test completed with zero cohort errors, zero malformed selected paths, zero
selected-source frames missing navigation metadata, and 100% coverage of all
mandatory endpoint-anchor instances. No Gaussian or RGB rendering was run.

## Scope

- Corpora: `formal_fast_v1`, `formal_slow_v1`, and
  `formal_social_slow_v1` on `pathGen_lxh`.
- Sampling: one deterministically ranked nonempty scene from each of 13
  corpus/mission-family groups, up to 10 complete source paths per scene.
- Seed: `20260911`.
- Configured average scored-POI floor: `0`.
- Sample: 127 complete paths and 31,160 source frames.
- Path lengths: mean 245.35, median 301, P25 143, P75 301, P90 366,
  P95/P99 381, minimum 66, maximum 381 frames.
- Temporal package: exactly 32 past frames plus current, no future frames;
  `target_window_index=32`.

## Selection and Retention

| Measurement | Result |
| --- | ---: |
| Scored event-aware POIs | 2,446 |
| Mean scored POIs/path | 19.26 |
| Mandatory anchors | 475 |
| Selected centers / independent windows | 2,921 |
| Independent 33-frame occurrences | 96,393 |
| Independent window/source volume | 309.35% |
| Unique retained source frames | 26,817 |
| Unique source-frame retention | 86.06% |
| Endpoint coverage | 100.00% |

The roughly three-source-equivalent independent-window volume confirms the
overlap observed in the full survey. Event-aware selection should therefore be
paired with render-once caching or source-frame references; this is a rendering
optimization and does not require deleting semantic representatives.

## Action Coverage

| Population | Move | Stop | Left | Right |
| --- | ---: | ---: | ---: | ---: |
| All source frames | 75.91% | 10.72% | 6.94% | 6.43% |
| Scored POI centers | 52.45% | 9.93% | 19.09% | 18.52% |
| Independent windows | 65.22% | 14.87% | 10.09% | 9.82% |
| Unique retained frames | 73.75% | 11.32% | 7.76% | 7.17% |

The selected centers substantially increase distinct left/right decisions,
while the causal windows preserve forward approach context and approximately
the source stop prevalence.

## Episode Coverage

Every explicitly guaranteed signal reached 100% coverage by any selected
center: collision avoidance, human-interaction decisions, instruction-section
boundaries, instruction turns, planned-execution mismatches, room transitions,
route deviations, semantic stops, social-law decisions, target-human
visibility, and traffic waits. Mission endpoints also reached 100% through the
mandatory-anchor pool.

`relevant_human_visible` is recorded and scored but is not a guaranteed signal;
119 of 132 eligible episodes (90.15%) were covered by any center.

Scored POIs excluding mandatory anchors covered:

- 100% of instruction-turn and room-transition episodes;
- 97.35% of instruction-section boundaries and 97.33% of stop episodes;
- 95.59% of route deviations;
- 92.50% of execution mismatches and 92.02% of collision avoidance;
- 90.48% of target-human-visible episodes;
- 80.30% of relevant-human-visible episodes.

Lower scored-only coverage for endpoint-aligned human interactions, social-law
decisions, mission endpoints, and traffic waits is expected when the same center
belongs to the mandatory-anchor pool. Their any-center coverage is 100%.

## Validation and Visual QA

- Focused selector tests: 36 passed.
- Maintained Datagen suite: 154 passed, 1 skipped.
- The default-policy test verifies that a zero-POI floor selects the two
  detected collision-avoidance episodes instead of all eligible frames.
- All 13 representative JSON files report
  `semantic_episode_policy=guarantee_representative`, a zero configured floor,
  and policy v0.6.
- All 283 representative chunks have 33 source indices, end at the target,
  contain no future frames, and use `target_window_index=32`.
- All 13 mission-family BEV overlays and GIFs were generated. Each GIF is
  animated and 900 by 800 pixels.
- Global action, navigation-signal, episode-coverage, and coverage-volume plots
  were generated and visually inspected.
- Single-policy title/legend overlap found during visual QA was fixed before
  regenerating the final artifacts.

## Artifacts

Local root:
`out/poi_policy_survey_20260911/event_aware_pure_default_limited_20260911`

Remote root:
`/private_lxh/dongjk/navdata/mass_generation_runs/poi_policy_survey_20260911/event_aware_pure_default_limited_20260911`

The roots contain the JSON, CSV, and Markdown summaries. The remote root also
contains the compressed renderer-independent
`selection_manifests/event_aware.jsonl.gz` selection manifest. The local
`visualizations` directory contains four statistical plots, 13 BEV overlays,
13 GIFs, 13 instruction/metadata companions, and a JSON index.
