#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.run_config import prepare_render_run_from_config_path  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and preflight a self-service MassGen render run from a JSON config."
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the generated render manifest and optional summary JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight only; do not write files. This is the default unless --write is set.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Alias for --dry-run.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the run summary JSON to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.write and (args.dry_run or args.preflight_only):
        print("--write cannot be combined with --dry-run or --preflight-only", file=sys.stderr)
        return 2
    result = prepare_render_run_from_config_path(args.config_json, write_outputs=bool(args.write))
    summary = result["summary"]
    if args.summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Prepared {summary['job_count']} job(s) for scenario {summary['scenario_id']} "
            f"with status={summary['status']}."
        )
        print(f"Manifest: {result['manifest_json']}")
        if result.get("summary_json"):
            print(f"Summary: {result['summary_json']}")
        if summary["preflight"]["errors"]:
            print("Errors:")
            for error in summary["preflight"]["errors"]:
                print(f"- {error}")
        if summary["warnings"]:
            print(f"Warnings: {len(summary['warnings'])}")
            for warning in summary["warnings"][:10]:
                print(f"- {warning}")
    return 1 if summary["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
