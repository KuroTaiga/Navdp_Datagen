# Script Layout

First-party helper scripts are grouped by domain. Keep repo-root scripts limited
to active renderers and compatibility launchers.

## Domains

- `actions/`: actor assignment and action JSON export/conversion.
- `analysis/`: input/render reporting, bottleneck diagnostics, and output eval.
- `datasets/`: dataset materialization and format conversion.
- `legacy_graphdeco/`: original GraphDeCo train/render/eval utilities kept for
  compatibility, not MassGen rendering.
- `massgen/`: Pathplanner scenario to MassGen render-manifest tools.
- `media/`: video frame extraction, mosaics, grids, and side-by-side media.
- `render/`: render-specific utilities outside the primary root renderers.
- `render/assets/`: robot/asset conversion and GLB overlay tools.
- `render/compare/`: render comparison utilities.
- `render/views/`: view/preview/verification renderers.
- `smoke/`: small manual smoke tests and demo launchers.
- `storage/`: storage, sync, archive, and cleanup helpers.

## Rules

- Add new scripts under the relevant domain folder, not repo root.
- Put reusable logic in `utils/` or a future package module before adding another
  wrapper.
- Avoid hard-coded local machine paths. Use CLI arguments or environment
  variables and fail clearly when required paths are missing.
- Keep root shell wrappers as compatibility shims only until a unified launcher
  replaces them.
