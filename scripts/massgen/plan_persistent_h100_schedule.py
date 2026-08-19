#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run a scene/resource-aware persistent H100 render schedule from "
            "a MassGen render_plan.json."
        )
    )
    parser.add_argument("--render-plan-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan_payload = json.loads(args.render_plan_json.read_text(encoding="utf-8"))
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
    payload = schedule.to_json_dict()
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


if __name__ == "__main__":
    raise SystemExit(main())
