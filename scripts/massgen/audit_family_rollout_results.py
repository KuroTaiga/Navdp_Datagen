#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MassGen family rollout result consistency.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--max-camera-label-xy-diff-m", type=float, default=0.05)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _family_record(family_dir: Path, *, max_camera_label_xy_diff_m: float) -> dict[str, Any]:
    manifest = _load_json(family_dir / "render_manifest.json")
    source = manifest.get("source", {})
    if not isinstance(source, dict):
        source = {}
    jobs = [job for job in manifest.get("jobs", []) if isinstance(job, dict)]
    job = jobs[0] if jobs else {}
    scenario_id = str(source.get("scenario_id") or "")
    scene_id = str(source.get("scene_id") or "")
    job_id = str(job.get("job_id") or "")
    label_paths = sorted((family_dir / "render_inputs").glob("**/label_paths/*.json"))
    camera_jsons = sorted((family_dir / "renders").glob("**/*_camera.json"))
    videos = sorted((family_dir / "renders").glob("**/*.mp4"))
    bevs = sorted(family_dir.glob("*_bev_trajectory.*"))

    label_ok = False
    camera_ok = False
    max_xy_diff_m: float | None = None
    frame_count: int | None = None
    if label_paths:
        label = _load_json(label_paths[0])
        label_ok = str(label.get("ins_id") or "") == job_id and str(label.get("scene_id") or "") == scene_id
    if label_paths and camera_jsons:
        label = _load_json(label_paths[0])
        camera = _load_json(camera_jsons[0])
        world = label.get("path", {}).get("raster_world", [])
        frames = camera.get("frames", [])
        if isinstance(world, list) and isinstance(frames, list):
            frame_count = len(frames)
            diffs: list[float] = []
            for world_item, frame_item in zip(world, frames):
                if not isinstance(world_item, dict) or not isinstance(frame_item, dict):
                    continue
                center = frame_item.get("camera_center_world")
                if not isinstance(center, list) or len(center) < 2:
                    continue
                dx = float(center[0]) - float(world_item.get("x", 0.0) or 0.0)
                dy = float(center[1]) - float(world_item.get("y", 0.0) or 0.0)
                diffs.append(math.hypot(dx, dy))
            max_xy_diff_m = max(diffs) if diffs else None
            camera_ok = (
                str(camera.get("label") or "") == job_id
                and len(frames) == len(world)
                and max_xy_diff_m is not None
                and max_xy_diff_m <= max_camera_label_xy_diff_m
            )

    video_ok = bool(videos) and all(job_id in video.name for video in videos)
    bev_ok = all(scenario_id in bev.name for bev in bevs)
    ok = bool(label_ok and camera_ok and video_ok and bev_ok)
    return {
        "family": family_dir.name,
        "ok": ok,
        "scenario_id": scenario_id,
        "scene_id": scene_id,
        "job_id": job_id,
        "frame_count": frame_count,
        "label_ok": label_ok,
        "camera_ok": camera_ok,
        "max_camera_label_xy_diff_m": max_xy_diff_m,
        "video_ok": video_ok,
        "bev_ok": bev_ok,
        "bev_files": [bev.name for bev in bevs],
        "video_files": [video.name for video in videos],
        "label_path": str(label_paths[0]) if label_paths else None,
        "camera_json": str(camera_jsons[0]) if camera_jsons else None,
    }


def main() -> int:
    args = _parse_args()
    results_root = args.results_root.expanduser().resolve()
    families_root = results_root / "families"
    records = [
        _family_record(path, max_camera_label_xy_diff_m=float(args.max_camera_label_xy_diff_m))
        for path in sorted(families_root.iterdir())
        if path.is_dir()
    ]
    report = {
        "results_root": str(results_root),
        "family_count": len(records),
        "success_count": sum(1 for record in records if record["ok"]),
        "status": "success" if records and all(record["ok"] for record in records) else "failed",
        "max_camera_label_xy_diff_m": float(args.max_camera_label_xy_diff_m),
        "families": records,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for record in records:
        diff = record["max_camera_label_xy_diff_m"]
        diff_text = "None" if diff is None else f"{float(diff):.4f}"
        print(
            f"{record['family']}: ok={record['ok']} frames={record['frame_count']} "
            f"label={record['label_ok']} camera={record['camera_ok']} "
            f"max_xy_diff_m={diff_text} video={record['video_ok']} bev={record['bev_ok']}"
        )
    print(f"summary_status {report['status']} {report['success_count']} / {report['family_count']}")
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
