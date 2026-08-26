# Human Avatar Remote Assets

Last checked: 2026-08-24

## Hosts

- Source PC: `4090_Sun` (`lenovo@192.168.152.171`)
- Destination platform: `pathGen_lxh` (`root@10.127.48.252:31115`)
- H100 formal-test platform: `envtest` (`root@10.127.48.252:31117`)
- Key auth: local `~/.ssh/id_ed25519.pub` is already present on `4090_Sun`.

Plaintext fallback passwords for the source PC are intentionally not stored in
this repository. As of 2026-08-23, password auth was tested once against
`4090_Sun` and key auth remains the normal path.

## 20260811 STMC/Kimodo Action Batch

Destination root on `pathGen_lxh`:

```text
/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions
```

Source and destination mapping:

| Set | Source on `4090_Sun` | Destination on `pathGen_lxh` |
| --- | --- | --- |
| Kimodo prompt | `/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/prompts/expanded_actions.txt` | `kimodo/prompts/expanded_actions.txt` |
| Kimodo motion data | `/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expandedforkimodo/kimodo` | `kimodo/motionjson/` |
| Kimodo rendered outputs | `/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_k` | `kimodo/outputs/` |
| STMC prompt | `/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/prompts/stmc_more_actions.txt` | `stmc/prompts/stmc_more_actions.txt` |
| STMC motion data | `/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expanded/stmc` | `stmc/motionjson/` |
| STMC rendered outputs | `/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_s` | `stmc/outputs/` |

Grouped action folders on `pathGen_lxh`:

```text
/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/grouped_actions/use_default
/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/grouped_actions/contextual
/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/grouped_actions/reject_default
```

Each group contains both `kimodo/` and `stmc/`, with `motionjson/`,
`outputs/`, and `previews/` symlinks. The group manifests are under:

```text
/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/grouped_actions/manifests
```

Current curated seed counts:

| Group | Kimodo | STMC | Total |
| --- | ---: | ---: | ---: |
| `use_default` | 20 | 27 | 47 |
| `contextual` | 10 | 10 | 20 |
| `reject_default` | 10 | 23 | 33 |

## H100 Formal-Test Action Pool

Destination on `envtest`:

```text
/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/grouped_actions/use_default_no_waving
```

This group is generated from the platform-local full asset tree and keeps only
the green/default rows from `intern_action_decision_seed.csv`, excluding waving
actions by prompt text. Rebuild it with:

```sh
python scripts/storage/group_human_avatar_actions_on_platform.py \
  --host envtest \
  --group-name use_default_no_waving
```

As checked on 2026-08-24:

| Set | Kimodo | STMC | Total |
| --- | ---: | ---: | ---: |
| Selected action rows | 18 | 26 | 44 |
| Motion JSON links | 18 | 26 | 44 |
| Rendered output links | 18 | 24 | 42 |
| Preview links | 18 | 22 | 40 |

Excluded wave actions: `K037`, `K038`, `S015`.

Known H100 asset gaps in this pool:

- `S000` and `S009` have motion JSON but no rendered output directory on the
  current H100 `/team` asset root.
- `S005` and `S018` have rendered output directories but no `preview.mp4`.

## 5880 Default-Action Copy

Destination on `5880host`:

```text
/mnt/DATA/dongjk/navdp_data/human_avatars/20260811_stmc_kimodo_new_actions/use_default
```

Before copying, `/mnt/DATA/pjc_temp` was moved to `/mnt/DATA1/pjc_temp`.
`/mnt/DATA/pjc_temp` is now a symlink to `/mnt/DATA1/pjc_temp`. The
default-action copy controller waited until `/mnt/DATA` had at least 220G
available, then started copying only `use_default` actions from `4090_Sun`.

Space notes from 2026-08-23:

- `/mnt/DATA/tmp*` entries are tiny scratch directories, only KB-scale.
- `/mnt/DATA/dongjk/navdp_data/human_gs_source` is about 6.3G, so it is not
  worth moving unless the copy still needs a small amount of extra space.
- `rsync` completed for `/mnt/DATA1/pjc_temp`, but 18G of source-side leftovers
  could not be deleted by the `dongjk` login because they are owned by
  `lenovo:lenovo` under owner-writable-only directories. They were moved aside
  to `/mnt/DATA/pjc_temp_leftover_lenovo_owned_20260823`; cleanup requires the
  `lenovo` login or sudo.
