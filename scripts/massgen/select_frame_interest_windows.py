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
    FrameSelectionConfig,
    apply_frame_selection_to_manifest,
    load_json,
    select_frame_interest_windows_from_paths,
    write_json,
)


def _parse_ratio(value: str) -> tuple[str, float]:
    key, sep, raw_number = str(value).partition("=")
    if sep != "=" or not key.strip():
        raise argparse.ArgumentTypeError("ratios must use name=value")
    try:
        number = float(raw_number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ratio value: {value}") from exc
    if number < 0.0:
        raise argparse.ArgumentTypeError("ratios must be non-negative")
    return key.strip(), number


def _ratio_map(values: list[tuple[str, float]] | None, defaults: dict[str, float] | None = None) -> dict[str, float]:
    out = dict(defaults or {})
    for key, value in values or []:
        out[str(key)] = float(value)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select navigation frame-of-interest windows and optionally emit a "
            "render manifest that renders only those 32+1+32 frame windows."
        )
    )
    parser.add_argument("--manifest-json", type=Path, action="append", required=True)
    parser.add_argument("--output-selection-json", type=Path, required=True)
    parser.add_argument(
        "--output-render-manifest-json",
        type=Path,
        default=None,
        help="Optional selected-window render manifest. Supported when one --manifest-json is provided.",
    )
    parser.add_argument("--target-count", type=int, default=0, help="0 means select every eligible target frame.")
    parser.add_argument("--past-frames", type=int, default=DEFAULT_PAST_FRAMES)
    parser.add_argument("--future-frames", type=int, default=DEFAULT_FUTURE_FRAMES)
    parser.add_argument("--edge-window-policy", choices=["reject", "clamp"], default="reject")
    parser.add_argument("--min-target-spacing-frames", type=int, default=16)
    parser.add_argument("--max-targets-per-job", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help="Only consider jobs belonging to this mission family. Repeatable.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--densify-frame-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate sparse trajectory samples across integer source frame ids.",
    )
    parser.add_argument(
        "--preserve-mission-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always select assigned mission/section completions and each trajectory endpoint.",
    )
    parser.add_argument("--endpoint-window-policy", choices=["reject", "clamp"], default="clamp")
    parser.add_argument(
        "--max-source-paths",
        type=int,
        default=0,
        help="Seeded random whole-path sample size; zero keeps all eligible paths.",
    )
    parser.add_argument(
        "--bucket-ratio",
        action="append",
        type=_parse_ratio,
        default=None,
        metavar="BUCKET=SHARE",
        help="Override adaptive interest-bucket ratios. Repeatable.",
    )
    parser.add_argument(
        "--action-ratio",
        action="append",
        type=_parse_ratio,
        default=None,
        metavar="ACTION=SHARE",
        help="Override secondary action ratios, e.g. stop=0.2 move=0.4.",
    )
    parser.add_argument("--action-deficit-weight", type=float, default=0.35)
    parser.add_argument("--json", action="store_true", help="Print the selection summary as JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output_render_manifest_json is not None and len(args.manifest_json) != 1:
        print(
            "--output-render-manifest-json currently requires exactly one --manifest-json; "
            "run the command once per source manifest for selected render manifests.",
            file=sys.stderr,
        )
        return 2

    config = FrameSelectionConfig(
        target_count=int(args.target_count),
        past_frames=int(args.past_frames),
        future_frames=int(args.future_frames),
        edge_window_policy=str(args.edge_window_policy),
        min_target_spacing_frames=int(args.min_target_spacing_frames),
        max_targets_per_job=int(args.max_targets_per_job),
        seed=int(args.seed),
        split=str(args.split),
        mission_families=tuple(str(item) for item in (args.family or [])),
        densify_frame_gaps=bool(args.densify_frame_gaps),
        max_source_paths=int(args.max_source_paths),
        preserve_mission_endpoints=bool(args.preserve_mission_endpoints),
        endpoint_window_policy=str(args.endpoint_window_policy),
        target_bucket_ratios=_ratio_map(args.bucket_ratio) if args.bucket_ratio else None,
        target_action_ratios=_ratio_map(args.action_ratio, DEFAULT_ACTION_RATIOS),
        action_deficit_weight=float(args.action_deficit_weight),
    )
    selection = select_frame_interest_windows_from_paths(args.manifest_json, config=config)
    selection["selection_path"] = str(args.output_selection_json)
    write_json(args.output_selection_json, selection)

    output_manifest = None
    if args.output_render_manifest_json is not None:
        source_manifest = load_json(args.manifest_json[0])
        selected_manifest = apply_frame_selection_to_manifest(
            source_manifest,
            selection,
            manifest_path=args.manifest_json[0],
        )
        write_json(args.output_render_manifest_json, selected_manifest)
        output_manifest = str(args.output_render_manifest_json)

    summary = {
        "selection_json": str(args.output_selection_json),
        "selected_render_manifest_json": output_manifest,
        "selected_target_count": selection["selection_summary"]["selected_target_count"],
        "requested_window_frame_renders": selection["selection_summary"]["requested_window_frame_renders"],
        "unique_source_render_frame_count": selection["selection_summary"]["unique_source_render_frame_count"],
        "selected_bucket_counts": selection["selection_summary"]["selected_bucket_counts"],
        "selected_action_counts": selection["selection_summary"]["selected_action_counts"],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "Selected {selected_target_count} target frame(s), "
            "{requested_window_frame_renders} requested window frame render(s), "
            "{unique_source_render_frame_count} unique source frame(s).".format(**summary)
        )
        print(f"Selection: {summary['selection_json']}")
        if output_manifest is not None:
            print(f"Selected render manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
