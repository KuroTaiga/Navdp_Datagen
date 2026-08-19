#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.persistent_scheduler import (  # noqa: E402
    GiB,
    ResourceCacheState,
    ResourceEstimates,
    ResourceRef,
    build_persistent_gpu_schedule,
)
from navdp_datagen.massgen.render_executor import (  # noqa: E402
    build_render_plans,
    load_render_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run a scene/resource-aware persistent H100 render schedule from "
            "a MassGen render_plan.json."
        )
    )
    parser.add_argument(
        "--render-plan-json",
        type=Path,
        default=None,
        help="Existing aggregate render_plan.json. Mutually exclusive with --package-root.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help="Smoke package root containing smoketest_package_index.json.",
    )
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=None,
        help=(
            "Root for package-derived render inputs/plans. Defaults to "
            "<output-json parent>/materialized_render_plans when possible."
        ),
    )
    parser.add_argument("--render-plan-output-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--max-renders", type=int, default=0)
    parser.add_argument("--renders-per-family-source-scene", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--render-script", type=Path, default=REPO_ROOT / "render_label_paths_telesim.py")
    parser.add_argument("--video-backend", default="cpu", choices=["cpu", "nvenc", "gpu"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimal-frames", type=int, default=None)
    parser.add_argument("--actor-gpu-resident", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-runtime-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gpu-id",
        action="append",
        default=None,
        help="GPU id to schedule onto. May be passed multiple times. Default: 0.",
    )
    parser.add_argument(
        "--max-items-per-chunk",
        type=int,
        default=0,
        help="Split compatible same-scene work into chunks of at most this many jobs.",
    )
    parser.add_argument("--scene-vram-gib", type=float, default=8.0)
    parser.add_argument("--human-avatar-vram-mib", type=float, default=512.0)
    parser.add_argument("--robot-asset-vram-mib", type=float, default=512.0)
    parser.add_argument("--actor-plan-ram-mib", type=float, default=64.0)
    parser.add_argument(
        "--simulate-cache-gib",
        type=float,
        default=None,
        help="If set, simulate chunk resource admission with this VRAM capacity per GPU.",
    )
    parser.add_argument(
        "--include-execution",
        action="store_true",
        help=(
            "Include original render plans and commands in each chunk so "
            "run_persistent_h100_schedule.py can execute the schedule."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (args.render_plan_json is None) == (args.package_root is None):
        print("Pass exactly one of --render-plan-json or --package-root.", file=sys.stderr)
        return 2
    if args.render_plan_json is not None:
        plan_payload = json.loads(args.render_plan_json.read_text(encoding="utf-8"))
    else:
        plan_payload = _build_render_plan_from_package(args)
        if args.render_plan_output_json is not None:
            args.render_plan_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.render_plan_output_json.write_text(
                json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    estimates = ResourceEstimates(
        scene_vram_bytes=int(float(args.scene_vram_gib) * GiB),
        human_avatar_vram_bytes=int(float(args.human_avatar_vram_mib) * 1024 * 1024),
        robot_asset_vram_bytes=int(float(args.robot_asset_vram_mib) * 1024 * 1024),
        actor_plan_ram_bytes=int(float(args.actor_plan_ram_mib) * 1024 * 1024),
    )
    schedule = build_persistent_gpu_schedule(
        plan_payload,
        gpu_ids=[str(item) for item in (args.gpu_id or ["0"])],
        max_items_per_chunk=int(args.max_items_per_chunk or 0),
        estimates=estimates,
    )
    payload = schedule.to_json_dict(include_execution=bool(args.include_execution))
    if args.simulate_cache_gib is not None:
        payload["cache_simulation"] = _simulate_cache(
            schedule.to_json_dict(),
            capacity_vram_bytes=int(float(args.simulate_cache_gib) * GiB),
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def _build_render_plan_from_package(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.expanduser().resolve()
    index_path = package_root / "smoketest_package_index.json"
    package_index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [entry for entry in package_index.get("entries", []) if isinstance(entry, Mapping)]
    materialized_root = _materialized_root(args, package_root=package_root)
    selected: list[Mapping[str, Any]] = []
    per_group_counts: dict[tuple[str, str, str], int] = {}
    for entry in entries:
        if not _passes_filters(entry, args.family, args.source, args.scene):
            continue
        group_key = (
            str(entry.get("family")),
            str(entry.get("source")),
            str(entry.get("scene")),
        )
        if int(args.renders_per_family_source_scene or 0) > 0:
            current = per_group_counts.get(group_key, 0)
            if current >= int(args.renders_per_family_source_scene):
                continue
            per_group_counts[group_key] = current + 1
        selected.append(entry)
        if int(args.max_renders or 0) > 0 and len(selected) >= int(args.max_renders):
            break

    all_plans: list[dict[str, Any]] = []
    entry_records: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(selected):
        manifest_path = Path(str(entry["render_manifest_json"]))
        if not manifest_path.is_absolute():
            manifest_path = package_root / manifest_path
        entry_root = materialized_root
        manifest = load_render_manifest(manifest_path)
        plan_payload = build_render_plans(
            manifest,
            manifest_path=manifest_path,
            output_root=entry_root,
            render_script=args.render_script,
            python_bin=str(args.python_bin),
            write_inputs=True,
            video_backend=str(args.video_backend),
            device=str(args.device),
            minimal_frames=args.minimal_frames,
            actor_gpu_resident=bool(args.actor_gpu_resident),
            actor_runtime_cache=bool(args.actor_runtime_cache),
        )
        entry_records.append(
            {
                "entry_index": entry_index,
                "family": entry.get("family"),
                "source": entry.get("source"),
                "scene": entry.get("scene"),
                "manifest_json": str(manifest_path),
                "output_root": str(entry_root),
                "plan_status": plan_payload.get("status"),
                "job_count": plan_payload.get("job_count"),
            }
        )
        for plan in plan_payload.get("plans", []):
            if isinstance(plan, Mapping):
                all_plans.append(dict(plan))

    return {
        "schema_version": "massgen_aggregate_render_plan.v1",
        "source": "plan_persistent_h100_schedule.py",
        "package_root": str(package_root),
        "materialized_root": str(materialized_root),
        "selected_entry_count": len(selected),
        "job_count": len(all_plans),
        "status": "blocked" if any(plan.get("blockers") for plan in all_plans) else "ready",
        "entries": entry_records,
        "plans": all_plans,
    }


def _simulate_cache(
    schedule_payload: dict[str, object],
    *,
    capacity_vram_bytes: int,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for assignment in schedule_payload.get("assignments", []):
        if not isinstance(assignment, dict):
            continue
        gpu_id = str(assignment.get("gpu_id"))
        cache = ResourceCacheState(capacity_vram_bytes=capacity_vram_bytes)
        operations: list[dict[str, object]] = []
        for chunk in assignment.get("chunks", []):
            if not isinstance(chunk, dict):
                continue
            resources = []
            for resource in chunk.get("resources", []):
                if not isinstance(resource, dict):
                    continue
                resources.append(
                    ResourceRef(
                        kind=str(resource.get("kind")),
                        key=str(resource.get("key")),
                        path=resource.get("path") if isinstance(resource.get("path"), str) else None,
                        estimated_vram_bytes=int(resource.get("estimated_vram_bytes") or 0),
                        estimated_ram_bytes=int(resource.get("estimated_ram_bytes") or 0),
                        shareable_via_cuda_ipc=bool(resource.get("shareable_via_cuda_ipc", True)),
                    )
                )
            acquired = cache.acquire(resources)
            released = cache.release(resources)
            operations.extend(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "op": op.op,
                    "resource": op.resource.stable_id,
                    "used_vram_bytes": op.used_vram_bytes,
                }
                for op in (*acquired, *released)
            )
        out[gpu_id] = {
            "capacity_vram_bytes": capacity_vram_bytes,
            "final_used_vram_bytes": cache.used_vram_bytes,
            "operations": operations,
        }
    return out


def _materialized_root(args: argparse.Namespace, *, package_root: Path) -> Path:
    if args.materialized_root is not None:
        return args.materialized_root.expanduser().resolve()
    if args.output_json is not None:
        return args.output_json.expanduser().resolve().parent / "materialized_render_plans"
    return package_root / "persistent_h100_schedule_inputs"


def _passes_filters(
    entry: Mapping[str, Any],
    families: list[str] | None,
    sources: list[str] | None,
    scenes: list[str] | None,
) -> bool:
    if families and str(entry.get("family")) not in set(families):
        return False
    if sources and str(entry.get("source")) not in set(sources):
        return False
    if scenes and str(entry.get("scene")) not in set(scenes):
        return False
    return True

if __name__ == "__main__":
    raise SystemExit(main())
