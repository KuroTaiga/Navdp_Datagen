#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize natural/capped frame lengths for a persistent H100 aggregate render plan."
    )
    parser.add_argument("--aggregate-render-plan-json", type=Path, required=True)
    parser.add_argument("--schedule-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _option_after(command: list[str], option: str, default: str | None = None) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return default


def _base_mission_id(job_id: str) -> str:
    return re.sub(r"__view_[^/]+$", "", job_id)


def _summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values = sorted(values)

    def pct(q: float) -> float:
        if len(values) == 1:
            return values[0]
        index = (len(values) - 1) * q
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return values[low]
        return values[low] * (high - index) + values[high] * (index - low)

    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": max(values),
        "sum": sum(values),
    }


def _path_frame_count(label_path: str | None) -> int:
    if not label_path:
        return 0
    try:
        payload = json.loads(Path(label_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    candidates: list[int] = []
    path = payload.get("path")
    if isinstance(path, Mapping):
        for key in ("raster_world", "raster_pixel", "points", "positions", "poses", "trajectory"):
            value = path.get(key)
            if isinstance(value, list):
                candidates.append(len(value))
    for key in ("frames", "states", "trajectory"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.append(len(value))
    return max(candidates) if candidates else 0


def _entry_lookup(plan_payload: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    family_by_base: dict[str, str] = {}
    source_scene_by_base: dict[str, tuple[str, str]] = {}
    for entry in plan_payload.get("entries", []):
        if not isinstance(entry, Mapping):
            continue
        manifest = Path(str(entry.get("manifest_json") or ""))
        name = manifest.name
        base = name[: -len(".render_manifest.json")] if name.endswith(".render_manifest.json") else manifest.stem
        family_by_base[base] = str(entry.get("family") or "unknown")
        source_scene_by_base[base] = (str(entry.get("source") or "unknown"), str(entry.get("scene") or ""))
    return family_by_base, source_scene_by_base


def _job_to_chunk(schedule_json: Path | None) -> dict[str, str]:
    if schedule_json is None or not schedule_json.is_file():
        return {}
    schedule = _load_json(schedule_json)
    out: dict[str, str] = {}
    for assignment in schedule.get("assignments", []):
        if not isinstance(assignment, Mapping):
            continue
        for chunk in assignment.get("chunks", []):
            if not isinstance(chunk, Mapping):
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            for job_id in chunk.get("job_ids", []):
                out[str(job_id)] = chunk_id
    return out


def main() -> int:
    args = _parse_args()
    plan_payload = _load_json(args.aggregate_render_plan_json.expanduser().resolve())
    family_by_base, source_scene_by_base = _entry_lookup(plan_payload)
    chunk_by_job = _job_to_chunk(args.schedule_json.expanduser().resolve() if args.schedule_json else None)

    records: list[dict[str, Any]] = []
    for plan in plan_payload.get("plans", []):
        if not isinstance(plan, Mapping):
            continue
        job_id = str(plan.get("job_id") or "")
        base_id = _base_mission_id(job_id)
        command = [str(item) for item in plan.get("command", [])]
        natural_frames = _path_frame_count(plan.get("label_path") if isinstance(plan.get("label_path"), str) else None)
        minimal = _option_after(command, "--minimal-frames")
        capped_frames = min(natural_frames, int(minimal)) if minimal else natural_frames
        fps = float(_option_after(command, "--video-fps", "10") or 10.0)
        family = family_by_base.get(base_id)
        if family is None:
            families = plan.get("mission_families")
            family = "+".join(str(item) for item in families) if isinstance(families, list) else "unknown"
        source, source_scene = source_scene_by_base.get(base_id, ("unknown", ""))
        records.append(
            {
                "job_id": job_id,
                "mission_id": base_id,
                "family": family,
                "source": source,
                "scene": str(plan.get("scene_id") or source_scene),
                "chunk_id": chunk_by_job.get(job_id),
                "natural_frames": natural_frames,
                "capped_frames": capped_frames,
                "fps": fps,
                "natural_video_sec": natural_frames / fps if fps else None,
                "capped_video_sec": capped_frames / fps if fps else None,
                "label_path": plan.get("label_path"),
            }
        )

    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({record["family"] for record in records}):
        subset = [record for record in records if record["family"] == family]
        by_family[family] = {
            "render_track_count": len(subset),
            "mission_count": len({record["mission_id"] for record in subset}),
            "scene_count": len({record["scene"] for record in subset}),
            "source_count": len({record["source"] for record in subset}),
            "natural_frames": _summarize([float(record["natural_frames"]) for record in subset]),
            "capped_frames": _summarize([float(record["capped_frames"]) for record in subset]),
            "natural_video_sec": _summarize([float(record["natural_video_sec"] or 0.0) for record in subset]),
            "capped_video_sec": _summarize([float(record["capped_video_sec"] or 0.0) for record in subset]),
        }

    chunk_stats: dict[str, dict[str, Any]] = {}
    chunk_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        chunk_id = str(record.get("chunk_id") or "")
        chunk_counts[chunk_id]["jobs"] += 1
        chunk_counts[chunk_id]["natural_frames"] += int(record["natural_frames"])
        chunk_counts[chunk_id]["capped_frames"] += int(record["capped_frames"])
        chunk_counts[chunk_id][f"family:{record['family']}"] += 1
    for chunk_id, counts in chunk_counts.items():
        chunk_stats[chunk_id] = {
            "jobs": int(counts["jobs"]),
            "natural_frames": int(counts["natural_frames"]),
            "capped_frames": int(counts["capped_frames"]),
            "families": {
                key.split(":", 1)[1]: int(value)
                for key, value in counts.items()
                if key.startswith("family:")
            },
        }

    payload = {
        "schema_version": "massgen_full50_natural_length_projection.v1",
        "aggregate_render_plan_json": str(args.aggregate_render_plan_json),
        "schedule_json": str(args.schedule_json) if args.schedule_json else None,
        "job_count": len(records),
        "mission_count": len({record["mission_id"] for record in records}),
        "family_count": len(by_family),
        "scene_count": len({record["scene"] for record in records}),
        "source_count": len({record["source"] for record in records}),
        "overall": {
            "natural_frames": _summarize([float(record["natural_frames"]) for record in records]),
            "capped_frames": _summarize([float(record["capped_frames"]) for record in records]),
            "natural_video_sec": _summarize([float(record["natural_video_sec"] or 0.0) for record in records]),
            "capped_video_sec": _summarize([float(record["capped_video_sec"] or 0.0) for record in records]),
        },
        "by_family": by_family,
        "chunk_stats": chunk_stats,
        "records": records,
    }
    _write_json(args.output_json.expanduser().resolve(), payload)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
