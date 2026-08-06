#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.massgen_render_manifest import (  # noqa: E402
    DEFAULT_FPS,
    DEFAULT_RENDER_BACKEND,
    DEFAULT_ROBOT_GLB,
    scenario_file_to_render_manifest,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Pathplanner MassGen scenario JSON into a NavDP render manifest."
    )
    parser.add_argument("--scenario-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--action-catalog-json",
        type=Path,
        default=None,
        help="Optional Pathplanner action_codex.json used to resolve human action assets.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--render-backend", default=DEFAULT_RENDER_BACKEND)
    parser.add_argument("--default-robot-glb", default=DEFAULT_ROBOT_GLB)
    parser.add_argument(
        "--visibility-culling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable actor visibility culling in generated render jobs.",
    )
    parser.add_argument("--human-cull-margin-m", type=float, default=0.25)
    parser.add_argument("--robot-cull-margin-m", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = scenario_file_to_render_manifest(
        args.scenario_json,
        action_catalog_path=args.action_catalog_json,
        fps=float(args.fps),
        render_backend=str(args.render_backend),
        default_robot_glb=str(args.default_robot_glb),
        visibility_culling=bool(args.visibility_culling),
        human_cull_margin_m=float(args.human_cull_margin_m),
        robot_cull_margin_m=float(args.robot_cull_margin_m),
    )
    write_json(args.output_json, manifest)
    print(
        f"Wrote {args.output_json} with {len(manifest['jobs'])} job(s), "
        f"{len(manifest['actors']['humans'])} human(s), "
        f"{len(manifest['actors']['robots'])} robot(s)."
    )
    if manifest["warnings"]:
        print(f"Warnings: {len(manifest['warnings'])}")
        for warning in manifest["warnings"][:10]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
