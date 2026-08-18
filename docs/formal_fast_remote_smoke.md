# Formal Fast Remote Smoke

Use `scripts/massgen/formal_fast_remote_smoke.py` to sample the Pathplanner
formal mass-generation outputs on `pathGen_lxh` and verify the generated
structure before using the run as Datagen input.

Default remote root:

```text
/private_lxh/dongjk/navdata/mass_generation_runs/formal_fast_v1
```

Default behavior:

- connect to SSH host `pathGen_lxh`;
- enumerate every top-level mission family;
- for each family, sample one scene from each source:
  `CHINGMU_rescaled_1`, `CHINGMU_rescaled_2`, `CHINGMU_rescaled_3`, and
  `InteriorGS`;
- randomize scene order with seed `20260817`;
- pick the first sampled scene with exactly 500 mission JSONs and 500
  `_cornercase_metadata.json` files;
- download the selected scene's full `jsons/` tree plus the scene-level
  `mass_example_manifest.json`, `mass_generation_report.json`,
  `mass_generation_report.md`, and `mass_generation_progress.json`;
- write timing, count, download, and selected-scene manifests locally.

Run:

```bash
/Users/dongjk/miniconda3/bin/python3.13 scripts/massgen/formal_fast_remote_smoke.py \
  --output-root out/formal_fast_remote_smoke/formal_fast_v1_seed20260817 \
  --max-remote-elapsed-sec 60
```

Important outputs:

```text
out/formal_fast_remote_smoke/formal_fast_v1_seed20260817/
  smoke_manifest.json
  remote_summary.json
  download_report.json
  selected_scenes/
  scene_archives/
```

The smoke exits nonzero when any sampled family/source scene does not meet the
expected count, any download fails, or the optional remote elapsed threshold is
exceeded. The downloaded mission JSONs include the actual planned paths in
`robots[].trajectory` and the actor paths in `humans[].trajectory`. Use
`--download-mode examples` for the older small download of one mission JSON pair
per combination. Use `--sample-mode random` to check the first random scene
directly without skipping incomplete scenes. Use `--download-mode none` or
`--no-download` for a count-only probe.

The first count-only/example run on 2026-08-17 passed with 9 families, 36 sampled
family/source combinations, and remote probe time around 12 seconds. The full
scene JSON package mode is the preferred fixture mode for downstream structural
tests.
