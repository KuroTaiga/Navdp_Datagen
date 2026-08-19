from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


GiB = 1024**3
MiB = 1024**2


@dataclass(frozen=True)
class ResourceEstimates:
    """Conservative default resource estimates for scheduling before profiling."""

    scene_vram_bytes: int = 8 * GiB
    human_avatar_vram_bytes: int = 512 * MiB
    robot_asset_vram_bytes: int = 512 * MiB
    actor_plan_ram_bytes: int = 64 * MiB


@dataclass(frozen=True, order=True)
class ResourceRef:
    kind: str
    key: str
    path: str | None = None
    estimated_vram_bytes: int = 0
    estimated_ram_bytes: int = 0
    shareable_via_cuda_ipc: bool = True

    @property
    def stable_id(self) -> str:
        return f"{self.kind}:{self.key}"


@dataclass(frozen=True)
class RenderWorkItem:
    index: int
    job_id: str
    scene_id: str
    plan_payload: Mapping[str, Any] = field(repr=False, compare=False)
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    output_root: str | None
    label_path: str | None
    actor_plan_path: str | None
    human_actor_ids: tuple[str, ...]
    peer_robot_ids: tuple[str, ...]
    mission_families: tuple[str, ...]
    frame_count_hint: int
    resources: tuple[ResourceRef, ...]
    compatibility_key: tuple[str, ...]

    @property
    def estimated_vram_bytes(self) -> int:
        return _sum_unique_resource_bytes(self.resources, attr="estimated_vram_bytes")

    @property
    def estimated_ram_bytes(self) -> int:
        return _sum_unique_resource_bytes(self.resources, attr="estimated_ram_bytes")


@dataclass(frozen=True)
class SceneChunk:
    chunk_id: str
    scene_id: str
    items: tuple[RenderWorkItem, ...]
    resources: tuple[ResourceRef, ...]
    frame_count_hint: int
    estimated_vram_bytes: int
    estimated_ram_bytes: int


@dataclass(frozen=True)
class GpuAssignment:
    gpu_id: str
    chunks: tuple[SceneChunk, ...]

    @property
    def frame_count_hint(self) -> int:
        return sum(chunk.frame_count_hint for chunk in self.chunks)


@dataclass(frozen=True)
class PersistentGpuSchedule:
    assignments: tuple[GpuAssignment, ...]
    chunks: tuple[SceneChunk, ...]
    work_items: tuple[RenderWorkItem, ...]

    def to_json_dict(self, *, include_execution: bool = False) -> dict[str, Any]:
        return {
            "schema_version": "h100_persistent_schedule.v1",
            "work_item_count": len(self.work_items),
            "chunk_count": len(self.chunks),
            "includes_execution": bool(include_execution),
            "assignments": [
                {
                    "gpu_id": assignment.gpu_id,
                    "frame_count_hint": assignment.frame_count_hint,
                    "chunks": [
                        _chunk_to_json(chunk, include_execution=include_execution)
                        for chunk in assignment.chunks
                    ],
                }
                for assignment in self.assignments
            ],
        }


@dataclass(frozen=True)
class CacheOperation:
    op: str
    resource: ResourceRef
    used_vram_bytes: int


@dataclass
class ResourceCacheState:
    """Small deterministic VRAM cache model used by the future scheduler."""

    capacity_vram_bytes: int
    loaded: dict[str, ResourceRef] = field(default_factory=dict)
    lease_counts: dict[str, int] = field(default_factory=dict)
    last_used_tick: dict[str, int] = field(default_factory=dict)
    tick: int = 0

    @property
    def used_vram_bytes(self) -> int:
        return sum(ref.estimated_vram_bytes for ref in self.loaded.values())

    def acquire(self, resources: Sequence[ResourceRef]) -> list[CacheOperation]:
        operations: list[CacheOperation] = []
        for resource in _unique_resources(resources):
            operations.extend(self._ensure_loaded(resource))
            self.tick += 1
            self.lease_counts[resource.stable_id] = self.lease_counts.get(resource.stable_id, 0) + 1
            self.last_used_tick[resource.stable_id] = self.tick
            operations.append(CacheOperation("acquire", resource, self.used_vram_bytes))
        return operations

    def release(self, resources: Sequence[ResourceRef]) -> list[CacheOperation]:
        operations: list[CacheOperation] = []
        for resource in _unique_resources(resources):
            stable_id = resource.stable_id
            current = self.lease_counts.get(stable_id, 0)
            if current <= 0:
                continue
            next_count = current - 1
            if next_count:
                self.lease_counts[stable_id] = next_count
            else:
                self.lease_counts.pop(stable_id, None)
            self.tick += 1
            self.last_used_tick[stable_id] = self.tick
            operations.append(CacheOperation("release", resource, self.used_vram_bytes))
        return operations

    def _ensure_loaded(self, resource: ResourceRef) -> list[CacheOperation]:
        if resource.estimated_vram_bytes <= 0:
            return []
        if resource.stable_id in self.loaded:
            return [CacheOperation("reuse", resource, self.used_vram_bytes)]
        if resource.estimated_vram_bytes > self.capacity_vram_bytes:
            raise ValueError(
                f"resource {resource.stable_id} requires {resource.estimated_vram_bytes} bytes, "
                f"larger than cache capacity {self.capacity_vram_bytes}"
            )
        operations: list[CacheOperation] = []
        while self.used_vram_bytes + resource.estimated_vram_bytes > self.capacity_vram_bytes:
            evicted = self._evict_one()
            if evicted is None:
                raise RuntimeError(
                    f"cannot admit {resource.stable_id}; all eviction candidates are leased"
                )
            operations.append(evicted)
        self.loaded[resource.stable_id] = resource
        self.tick += 1
        self.last_used_tick[resource.stable_id] = self.tick
        operations.append(CacheOperation("load", resource, self.used_vram_bytes))
        return operations

    def _evict_one(self) -> CacheOperation | None:
        candidates = [
            stable_id
            for stable_id in self.loaded
            if self.lease_counts.get(stable_id, 0) <= 0
        ]
        if not candidates:
            return None
        stable_id = min(candidates, key=lambda item: self.last_used_tick.get(item, -1))
        resource = self.loaded.pop(stable_id)
        self.last_used_tick.pop(stable_id, None)
        return CacheOperation("evict", resource, self.used_vram_bytes)


def build_work_items_from_render_plan(
    plan_payload: Mapping[str, Any],
    *,
    estimates: ResourceEstimates | None = None,
    ready_only: bool = True,
) -> tuple[RenderWorkItem, ...]:
    estimates = estimates or ResourceEstimates()
    work_items: list[RenderWorkItem] = []
    for index, plan in enumerate(plan_payload.get("plans", [])):
        if not isinstance(plan, Mapping):
            continue
        if ready_only and str(plan.get("status")) != "ready":
            continue
        resources = _resources_for_plan(plan, estimates=estimates)
        command = tuple(str(item) for item in plan.get("command", []) if item is not None)
        work_items.append(
            RenderWorkItem(
                index=index,
                job_id=str(plan.get("job_id") or f"job_{index:06d}"),
                scene_id=str(plan.get("scene_id") or ""),
                plan_payload=dict(plan),
                command=command,
                env=tuple(
                    sorted(
                        (str(key), str(value))
                        for key, value in (plan.get("env") or {}).items()
                    )
                ),
                output_root=str((plan.get("metadata") or {}).get("output_root") or "") or None,
                label_path=_optional_str(plan.get("label_path")),
                actor_plan_path=_optional_str(plan.get("actor_plan_path")),
                human_actor_ids=tuple(str(item) for item in plan.get("human_actor_ids", []) if str(item)),
                peer_robot_ids=tuple(str(item) for item in plan.get("peer_robot_ids", []) if str(item)),
                mission_families=tuple(str(item) for item in plan.get("mission_families", []) if str(item)),
                frame_count_hint=_frame_count_hint(plan, command),
                resources=resources,
                compatibility_key=_compatibility_key(plan, command),
            )
        )
    return tuple(work_items)


def build_scene_chunks(
    work_items: Sequence[RenderWorkItem],
    *,
    max_items_per_chunk: int = 0,
) -> tuple[SceneChunk, ...]:
    grouped: dict[tuple[str, tuple[str, ...]], list[RenderWorkItem]] = {}
    for item in work_items:
        grouped.setdefault((item.scene_id, item.compatibility_key), []).append(item)
    chunks: list[SceneChunk] = []
    chunk_limit = max(0, int(max_items_per_chunk or 0))
    for group_index, key in enumerate(sorted(grouped)):
        items = grouped[key]
        slices = [items] if chunk_limit <= 0 else [
            items[index : index + chunk_limit]
            for index in range(0, len(items), chunk_limit)
        ]
        for chunk_index, item_slice in enumerate(slices):
            resources = _unique_resources(
                resource
                for item in item_slice
                for resource in item.resources
            )
            scene_id = key[0]
            chunk_id = f"{scene_id}_g{group_index:04d}_c{chunk_index:04d}"
            chunks.append(
                SceneChunk(
                    chunk_id=chunk_id,
                    scene_id=scene_id,
                    items=tuple(item_slice),
                    resources=resources,
                    frame_count_hint=sum(item.frame_count_hint for item in item_slice),
                    estimated_vram_bytes=_sum_unique_resource_bytes(resources, attr="estimated_vram_bytes"),
                    estimated_ram_bytes=_sum_unique_resource_bytes(resources, attr="estimated_ram_bytes"),
                )
            )
    return tuple(chunks)


def assign_chunks_to_gpus(
    chunks: Sequence[SceneChunk],
    *,
    gpu_ids: Sequence[str],
) -> tuple[GpuAssignment, ...]:
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    assignments: dict[str, list[SceneChunk]] = {str(gpu_id): [] for gpu_id in gpu_ids}
    load: dict[str, int] = {str(gpu_id): 0 for gpu_id in gpu_ids}
    scene_owner: dict[str, str] = {}
    for chunk in chunks:
        gpu_id = scene_owner.get(chunk.scene_id)
        if gpu_id is None:
            gpu_id = min(load, key=lambda item: (load[item], item))
            scene_owner[chunk.scene_id] = gpu_id
        assignments[gpu_id].append(chunk)
        load[gpu_id] += max(1, int(chunk.frame_count_hint))
    return tuple(
        GpuAssignment(gpu_id=str(gpu_id), chunks=tuple(assignments[str(gpu_id)]))
        for gpu_id in gpu_ids
    )


def build_persistent_gpu_schedule(
    plan_payload: Mapping[str, Any],
    *,
    gpu_ids: Sequence[str],
    max_items_per_chunk: int = 0,
    estimates: ResourceEstimates | None = None,
) -> PersistentGpuSchedule:
    work_items = build_work_items_from_render_plan(plan_payload, estimates=estimates)
    chunks = build_scene_chunks(work_items, max_items_per_chunk=max_items_per_chunk)
    assignments = assign_chunks_to_gpus(chunks, gpu_ids=gpu_ids)
    return PersistentGpuSchedule(
        assignments=assignments,
        chunks=chunks,
        work_items=work_items,
    )


def _resources_for_plan(
    plan: Mapping[str, Any],
    *,
    estimates: ResourceEstimates,
) -> tuple[ResourceRef, ...]:
    scene_id = str(plan.get("scene_id") or "")
    gaussian_model = _optional_str(plan.get("gaussian_model"))
    scene_root = _optional_str(plan.get("scene_root"))
    scene_key = "|".join(item for item in (scene_id, gaussian_model or scene_root or "") if item)
    resources: list[ResourceRef] = [
        ResourceRef(
            kind="scene",
            key=scene_key or scene_id,
            path=gaussian_model or scene_root,
            estimated_vram_bytes=estimates.scene_vram_bytes,
            estimated_ram_bytes=estimates.scene_vram_bytes,
            shareable_via_cuda_ipc=True,
        )
    ]
    actor_plan_path = _optional_str(plan.get("actor_plan_path"))
    if actor_plan_path:
        resources.append(
            ResourceRef(
                kind="actor_plan",
                key=actor_plan_path,
                path=actor_plan_path,
                estimated_ram_bytes=estimates.actor_plan_ram_bytes,
                shareable_via_cuda_ipc=False,
            )
        )
        resources.extend(_human_avatar_resources(actor_plan_path, estimates=estimates))
    else:
        for human_id in plan.get("human_actor_ids", []):
            resources.append(
                ResourceRef(
                    kind="human_avatar",
                    key=str(human_id),
                    estimated_vram_bytes=estimates.human_avatar_vram_bytes,
                    estimated_ram_bytes=estimates.human_avatar_vram_bytes,
                    shareable_via_cuda_ipc=True,
                )
            )
    for overlay in plan.get("robot_overlay_commands", []):
        if not isinstance(overlay, Mapping):
            continue
        robot_glb = _optional_str(overlay.get("robot_glb"))
        robot_urdf = _optional_str(overlay.get("robot_urdf"))
        actor_id = str(overlay.get("actor_id") or robot_glb or robot_urdf or "robot")
        key = "|".join(item for item in (actor_id, robot_glb or "", robot_urdf or "") if item)
        resources.append(
            ResourceRef(
                kind="robot_asset",
                key=key,
                path=robot_glb or robot_urdf,
                estimated_vram_bytes=estimates.robot_asset_vram_bytes,
                estimated_ram_bytes=estimates.robot_asset_vram_bytes,
                shareable_via_cuda_ipc=True,
            )
        )
    return _unique_resources(resources)


def _human_avatar_resources(
    actor_plan_path: str,
    *,
    estimates: ResourceEstimates,
) -> tuple[ResourceRef, ...]:
    path = Path(actor_plan_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    resources: list[ResourceRef] = []
    for actor in payload.get("actors", []):
        if not isinstance(actor, Mapping):
            continue
        actor_id = str(actor.get("actor_id") or "human")
        action = actor.get("action", {})
        if not isinstance(action, Mapping):
            action = {}
        asset = action.get("asset", {})
        if not isinstance(asset, Mapping):
            asset = {}
        asset_path = _optional_str(asset.get("ply_frame_dir")) or _optional_str(asset.get("source_ply_dir"))
        action_id = str(action.get("render_action_id") or action.get("action_sequence_id") or "")
        key = "|".join(item for item in (actor_id, action_id, asset_path or "") if item)
        resources.append(
            ResourceRef(
                kind="human_avatar",
                key=key,
                path=asset_path,
                estimated_vram_bytes=estimates.human_avatar_vram_bytes,
                estimated_ram_bytes=estimates.human_avatar_vram_bytes,
                shareable_via_cuda_ipc=True,
            )
        )
    return _unique_resources(resources)


def _frame_count_hint(plan: Mapping[str, Any], command: Sequence[str]) -> int:
    minimal = _single_option_value(command, "--minimal-frames")
    if minimal is not None:
        try:
            return max(1, int(minimal))
        except ValueError:
            pass
    label_path = _optional_str(plan.get("label_path"))
    if label_path:
        try:
            payload = json.loads(Path(label_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        path = payload.get("path", {}) if isinstance(payload, Mapping) else {}
        if isinstance(path, Mapping):
            for key in ("raster_world", "raster_pixel", "points"):
                value = path.get(key)
                if isinstance(value, list) and value:
                    return len(value)
    actor_plan_path = _optional_str(plan.get("actor_plan_path"))
    if actor_plan_path:
        try:
            payload = json.loads(Path(actor_plan_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        lengths = [
            len(actor.get("frames", []))
            for actor in payload.get("actors", [])
            if isinstance(actor, Mapping) and isinstance(actor.get("frames"), list)
        ]
        if lengths:
            return max(lengths)
    return 1


def _compatibility_key(plan: Mapping[str, Any], command: Sequence[str]) -> tuple[str, ...]:
    return (
        str(plan.get("scene_root") or ""),
        str(plan.get("gaussian_model") or ""),
        str(_single_option_value(command, "--device") or ""),
        str(_single_option_value(command, "--video-backend") or ""),
        "x".join(_option_values(command, "--resolution")),
        str(_single_option_value(command, "--fov-deg") or ""),
        str(_single_option_value(command, "--znear") or ""),
        str(_single_option_value(command, "--zfar") or ""),
        "depth" if "--save-depth-maps" in command else "no_depth",
        "rgb_frames" if "--rgb-frames" in command else "video_only",
    )


def _chunk_to_json(chunk: SceneChunk, *, include_execution: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "scene_id": chunk.scene_id,
        "job_ids": [item.job_id for item in chunk.items],
        "frame_count_hint": chunk.frame_count_hint,
        "estimated_vram_bytes": chunk.estimated_vram_bytes,
        "estimated_ram_bytes": chunk.estimated_ram_bytes,
        "resources": [
            {
                "kind": resource.kind,
                "key": resource.key,
                "path": resource.path,
                "estimated_vram_bytes": resource.estimated_vram_bytes,
                "estimated_ram_bytes": resource.estimated_ram_bytes,
                "shareable_via_cuda_ipc": resource.shareable_via_cuda_ipc,
            }
            for resource in chunk.resources
        ],
    }
    if include_execution:
        payload["work_items"] = [
            {
                "index": item.index,
                "job_id": item.job_id,
                "scene_id": item.scene_id,
                "command": list(item.command),
                "env": dict(item.env),
                "output_root": item.output_root,
                "label_path": item.label_path,
                "actor_plan_path": item.actor_plan_path,
                "human_actor_ids": list(item.human_actor_ids),
                "peer_robot_ids": list(item.peer_robot_ids),
                "mission_families": list(item.mission_families),
                "frame_count_hint": item.frame_count_hint,
            }
            for item in chunk.items
        ]
        payload["plans"] = [dict(item.plan_payload) for item in chunk.items]
    return payload


def _option_values(command: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == option:
            index += 1
            while index < len(command) and not str(command[index]).startswith("--"):
                values.append(str(command[index]))
                index += 1
            continue
        index += 1
    return values


def _single_option_value(command: Sequence[str], option: str) -> str | None:
    values = _option_values(command, option)
    return values[0] if values else None


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _unique_resources(resources: Sequence[ResourceRef] | Any) -> tuple[ResourceRef, ...]:
    by_id: dict[str, ResourceRef] = {}
    for resource in resources:
        if not isinstance(resource, ResourceRef):
            continue
        by_id.setdefault(resource.stable_id, resource)
    return tuple(sorted(by_id.values(), key=lambda item: item.stable_id))


def _sum_unique_resource_bytes(resources: Sequence[ResourceRef], *, attr: str) -> int:
    total = 0
    for resource in _unique_resources(resources):
        total += int(getattr(resource, attr))
    return total
