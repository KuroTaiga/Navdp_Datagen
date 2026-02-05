#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


ACTION_NAME = {
    0: "stop",
    1: "forward",
    2: "turn_left",
    3: "turn_right",
}


@dataclass(frozen=True)
class Counts:
    stop: int = 0
    forward: int = 0
    turn_left: int = 0
    turn_right: int = 0
    unknown: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "stop": self.stop,
            "forward": self.forward,
            "turn_left": self.turn_left,
            "turn_right": self.turn_right,
            "unknown": self.unknown,
        }

    def total(self) -> int:
        return self.stop + self.forward + self.turn_left + self.turn_right + self.unknown

    def add(self, other: "Counts") -> "Counts":
        return Counts(
            stop=self.stop + other.stop,
            forward=self.forward + other.forward,
            turn_left=self.turn_left + other.turn_left,
            turn_right=self.turn_right + other.turn_right,
            unknown=self.unknown + other.unknown,
        )


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(count) / float(total)


def counts_with_pct(counts: Counts) -> Dict[str, Dict[str, float]]:
    total = counts.total()
    pct = {
        "stop": _pct(counts.stop, total),
        "forward": _pct(counts.forward, total),
        "turn_left": _pct(counts.turn_left, total),
        "turn_right": _pct(counts.turn_right, total),
        "unknown": _pct(counts.unknown, total),
    }
    return {
        "counts": counts.to_dict(),
        "total": total,
        "pct": pct,
    }


def iter_scene_dirs(dataset_root: str) -> Iterable[os.DirEntry]:
    with os.scandir(dataset_root) as it:
        for entry in it:
            if entry.is_dir():
                yield entry


def list_action_jsons(scene_dir: str, pattern_suffix: str) -> List[str]:
    action_jsons: List[str] = []
    with os.scandir(scene_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if name.endswith(pattern_suffix):
                action_jsons.append(entry.path)
    action_jsons.sort()
    return action_jsons


def path_id_from_filename(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    if base.endswith(suffix):
        return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def count_actions_in_file(
    file_path: str,
) -> Tuple[Counts, Optional[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:  # noqa: BLE001
        return Counts(), f"failed_to_read_json: {e}"

    frames = payload.get("frames")
    if not isinstance(frames, list):
        return Counts(), "missing_or_invalid_frames"

    stop = forward = left = right = unknown = 0
    for frame_obj in frames:
        if not isinstance(frame_obj, dict):
            unknown += 1
            continue
        action = frame_obj.get("curr_action")
        if action == 0:
            stop += 1
        elif action == 1:
            forward += 1
        elif action == 2:
            left += 1
        elif action == 3:
            right += 1
        else:
            unknown += 1

    return (
        Counts(stop=stop, forward=forward, turn_left=left, turn_right=right, unknown=unknown),
        None,
    )


def print_summary_line(prefix: str, counts: Counts) -> None:
    total = counts.total()
    pct = counts_with_pct(counts)["pct"]
    print(
        f"{prefix} total={total} | "
        f"stop={counts.stop} ({pct['stop']:.2f}%) "
        f"forward={counts.forward} ({pct['forward']:.2f}%) "
        f"left={counts.turn_left} ({pct['turn_left']:.2f}%) "
        f"right={counts.turn_right} ({pct['turn_right']:.2f}%)"
        + (f" unknown={counts.unknown} ({pct['unknown']:.2f}%)" if counts.unknown else "")
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count curr_action values (0/1/2/3) in <path_id>_actions.json under each scene folder."
        )
    )
    p.add_argument(
        "dataset_root",
        nargs="?",
        default="./data2/0500_fpv",
        help="Dataset root containing scene subfolders (default: ./data2/0500_fpv).",
    )
    p.add_argument(
        "-o",
        "--output",
        default="./action_counts_0500.json",
        help="Output report JSON (default: ./action_counts_0500.json).",
    )
    p.add_argument(
        "--suffix",
        default="_actions.json",
        help="Only count files ending with this suffix (default: _actions.json).",
    )
    p.add_argument(
        "--no-per-path",
        action="store_true",
        help="Do not include per-path breakdown in the output JSON (faster/smaller).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    output_path = os.path.abspath(args.output)

    if not os.path.isdir(dataset_root):
        raise SystemExit(f"dataset_root not found or not a directory: {dataset_root}")

    report = {
        "dataset_root": dataset_root,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "action_map": {str(k): v for k, v in ACTION_NAME.items()},
        "scenes": {},
        "overall": {},
        "errors": [],
    }

    overall_counts = Counts()

    for scene_entry in sorted(iter_scene_dirs(dataset_root), key=lambda e: e.name):
        scene_name = scene_entry.name
        scene_dir = scene_entry.path
        action_jsons = list_action_jsons(scene_dir, args.suffix)

        scene_counts = Counts()
        scene_paths: Dict[str, Dict] = {}

        for json_path in action_jsons:
            path_id = path_id_from_filename(json_path, args.suffix)
            counts, err = count_actions_in_file(json_path)
            scene_counts = scene_counts.add(counts)
            overall_counts = overall_counts.add(counts)

            if err is not None:
                report["errors"].append({"file": json_path, "error": err})

            if not args.no_per_path:
                scene_paths[path_id] = {
                    "file": os.path.relpath(json_path, dataset_root),
                    **counts_with_pct(counts),
                }

        report["scenes"][scene_name] = {
            "num_action_json": len(action_jsons),
            **counts_with_pct(scene_counts),
            **({"paths": scene_paths} if not args.no_per_path else {}),
        }

    report["overall"] = counts_with_pct(overall_counts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")

    for scene_name, scene_info in report["scenes"].items():
        c = scene_info["counts"]
        scene_counts = Counts(
            stop=c["stop"],
            forward=c["forward"],
            turn_left=c["turn_left"],
            turn_right=c["turn_right"],
            unknown=c.get("unknown", 0),
        )
        print_summary_line(f"Scene {scene_name}:", scene_counts)

    print_summary_line("Overall:", overall_counts)

    if report["errors"]:
        print(f"Warnings: {len(report['errors'])} file(s) had errors. See report JSON for details.")

    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

