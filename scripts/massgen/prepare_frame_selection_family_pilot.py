#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.frame_selection import (  # noqa: E402
    DEFAULT_ACTION_RATIOS,
    DEFAULT_FUTURE_FRAMES,
    DEFAULT_PAST_FRAMES,
    DEFAULT_SEED,
    DEFAULT_SEMANTIC_EPISODE_POLICY,
)
from navdp_datagen.massgen.frame_selection_pilot import (  # noqa: E402
    MissionFamilyPilotConfig,
    SOCIAL_FAMILY_VARIANT_LAW_IDS,
    prepare_mission_family_pilot,
)
from utils.massgen_render_manifest import ACTIVE_MASS_MISSION_FAMILIES  # noqa: E402
from utils.massgen_render_manifest import (  # noqa: E402
    DEFAULT_FPS,
    scenario_file_to_render_manifest,
    write_json,
)


def _parse_ratio(value: str) -> tuple[str, float]:
    key, separator, raw_number = str(value).partition("=")
    if separator != "=" or not key.strip():
        raise argparse.ArgumentTypeError("ratios must use name=value")
    try:
        number = float(raw_number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ratio value: {value}") from exc
    if number < 0.0:
        raise argparse.ArgumentTypeError("ratios must be non-negative")
    return key.strip(), number


def _ratio_map(
    values: list[tuple[str, float]] | None,
    defaults: dict[str, float] | None = None,
) -> dict[str, float]:
    out = dict(defaults or {})
    for key, value in values or []:
        out[str(key)] = float(value)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, but do not execute, a frame-selection and render test "
            "matrix for every active MassGen mission family."
        )
    )
    parser.add_argument("--manifest-json", type=Path, action="append", default=None)
    parser.add_argument(
        "--scenario-json",
        type=Path,
        action="append",
        default=None,
        help="Pathplanner scenario JSON to convert before selection. Repeatable.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory searched recursively for --manifest-pattern. Repeatable.",
    )
    parser.add_argument("--manifest-pattern", default="*render_manifest*.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action-catalog-json", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help="Prepare only this mission family. Defaults to all active families.",
    )
    parser.add_argument(
        "--targets-per-family",
        type=int,
        default=0,
        help="Minimum scored-POI floor; zero uses pure event-aware selection.",
    )
    parser.add_argument("--past-frames", type=int, default=DEFAULT_PAST_FRAMES)
    parser.add_argument("--future-frames", type=int, default=DEFAULT_FUTURE_FRAMES)
    parser.add_argument("--edge-window-policy", choices=["reject", "clamp"], default="reject")
    parser.add_argument("--min-target-spacing-frames", type=int, default=16)
    parser.add_argument("--max-targets-per-job", type=int, default=0)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--render-repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Datagen checkout used by generated plan/execute commands.",
    )
    parser.add_argument(
        "--remote-sources-connected",
        action="store_true",
        help="Record that supplied filesystem manifests came from the connected databank or remote mount.",
    )
    parser.add_argument("--bucket-ratio", action="append", type=_parse_ratio, default=None)
    parser.add_argument("--action-ratio", action="append", type=_parse_ratio, default=None)
    parser.add_argument("--action-deficit-weight", type=float, default=0.35)
    parser.add_argument(
        "--densify-frame-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate sparse Pathplanner waypoints across integer renderer frame ids.",
    )
    parser.add_argument(
        "--preserve-mission-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always retain assigned sub-mission completions and final trajectory endpoints.",
    )
    parser.add_argument("--endpoint-window-policy", choices=["reject", "clamp"], default="clamp")
    parser.add_argument(
        "--semantic-episode-policy",
        choices=["none", "guarantee_representative"],
        default=DEFAULT_SEMANTIC_EPISODE_POLICY,
        help="Event-aware by default; use none only for fixed-budget comparisons.",
    )
    parser.add_argument(
        "--max-source-paths",
        type=int,
        default=0,
        help="Seeded random whole-path sample size per family; zero keeps all matching paths.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _manifest_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in (args.manifest_json or [])]
    for directory in args.manifest_dir or []:
        paths.extend(path for path in directory.rglob(str(args.manifest_pattern)) if path.is_file())
    return sorted({path.resolve() for path in paths}, key=str)


def _scenario_render_manifests(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output_root.resolve() / "source_render_manifests"
    paths: list[Path] = []
    for index, scenario_path in enumerate(args.scenario_json or []):
        scenario_path = scenario_path.resolve()
        manifest = scenario_file_to_render_manifest(
            scenario_path,
            action_catalog_path=args.action_catalog_json,
            fps=float(args.fps),
        )
        output_path = output_dir / f"{index:04d}_{scenario_path.stem}.render_manifest.json"
        write_json(output_path, manifest)
        paths.append(output_path)
    return paths


def main() -> int:
    args = _parse_args()
    default_families = (
        *ACTIVE_MASS_MISSION_FAMILIES,
        *SOCIAL_FAMILY_VARIANT_LAW_IDS,
    )
    families = tuple(str(item) for item in (args.family or default_families))
    config = MissionFamilyPilotConfig(
        mission_families=families,
        targets_per_family=int(args.targets_per_family),
        past_frames=int(args.past_frames),
        future_frames=int(args.future_frames),
        edge_window_policy=str(args.edge_window_policy),
        min_target_spacing_frames=int(args.min_target_spacing_frames),
        max_targets_per_job=int(args.max_targets_per_job),
        split=str(args.split),
        seed=int(args.seed),
        target_bucket_ratios=_ratio_map(args.bucket_ratio) if args.bucket_ratio else None,
        target_action_ratios=_ratio_map(args.action_ratio, DEFAULT_ACTION_RATIOS),
        action_deficit_weight=float(args.action_deficit_weight),
        python_bin=str(args.python_bin),
        remote_sources_connected=bool(args.remote_sources_connected),
        densify_frame_gaps=bool(args.densify_frame_gaps),
        max_source_paths=int(args.max_source_paths),
        preserve_mission_endpoints=bool(args.preserve_mission_endpoints),
        endpoint_window_policy=str(args.endpoint_window_policy),
        semantic_episode_policy=str(args.semantic_episode_policy),
        render_repo_root=args.render_repo_root,
    )
    manifest_paths = [*_manifest_paths(args), *_scenario_render_manifests(args)]
    plan = prepare_mission_family_pilot(
        manifest_paths,
        output_root=args.output_root,
        config=config,
    )
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Prepared {plan['summary']['family_count']} mission-family pilot entries: "
            f"{plan['status']}."
        )
        print(f"Pilot plan: {plan['pilot_plan_path']}")
        for record in plan["families"]:
            print(
                f"- {record['mission_family']}: {record['status']} "
                f"({record['selected_interest_target_count']}/{record['requested_target_count']} POIs + "
                f"{record['mandatory_anchor_count']} mandatory anchors)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
