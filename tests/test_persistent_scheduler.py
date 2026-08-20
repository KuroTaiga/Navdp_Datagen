from __future__ import annotations

import json
from pathlib import Path

import pytest

from navdp_datagen.massgen.persistent_scheduler import (
    ResourceCacheState,
    ResourceEstimates,
    ResourceRef,
    assign_chunks_to_gpus,
    build_persistent_gpu_schedule,
    build_scene_chunks,
    build_work_items_from_render_plan,
)


def _write_actor_plan(tmp_path: Path, *, actor_id: str = "human_a") -> Path:
    asset_dir = tmp_path / "avatars" / actor_id / "walk"
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{actor_id}_actors.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "massgen_actor_bundle.v1",
                "actors": [
                    {
                        "actor_id": actor_id,
                        "action": {
                            "render_action_id": "walk",
                            "asset": {"ply_frame_dir": str(asset_dir)},
                        },
                        "frames": [{"frame": index} for index in range(5)],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_label_path(tmp_path: Path, *, job_id: str, frames: int = 7) -> Path:
    path = tmp_path / f"{job_id}_label.json"
    path.write_text(
        json.dumps({"path": {"raster_world": [{"x": index, "y": 0.0} for index in range(frames)]}}),
        encoding="utf-8",
    )
    return path


def _plan(
    tmp_path: Path,
    *,
    job_id: str,
    scene_id: str,
    minimal_frames: int | None = None,
    actor_id: str = "human_a",
    with_robot: bool = False,
) -> dict[str, object]:
    actor_plan = _write_actor_plan(tmp_path, actor_id=actor_id)
    label_path = _write_label_path(tmp_path, job_id=job_id)
    command = [
        "python",
        "render_label_paths_telesim.py",
        "--scene",
        scene_id,
        "--resolution",
        "640",
        "480",
        "--fov-deg",
        "75",
        "--znear",
        "0.01",
        "--zfar",
        "100",
        "--video-backend",
        "cpu",
        "--device",
        "cuda",
        "--save-depth-maps",
        "--rgb-frames",
    ]
    if minimal_frames is not None:
        command.extend(["--minimal-frames", str(minimal_frames)])
    robot_overlay_commands = []
    peer_robot_ids = []
    if with_robot:
        robot_glb = tmp_path / "robots" / "g1.glb"
        robot_urdf = tmp_path / "robots" / "g1.urdf"
        robot_glb.parent.mkdir(parents=True, exist_ok=True)
        robot_glb.write_bytes(b"glb")
        robot_urdf.write_text("<robot />", encoding="utf-8")
        peer_robot_ids = ["robot_b"]
        robot_overlay_commands = [
            {
                "actor_id": "robot_b",
                "robot_glb": str(robot_glb),
                "robot_urdf": str(robot_urdf),
            }
        ]
    return {
        "status": "ready",
        "job_id": job_id,
        "scene_id": scene_id,
        "scene_root": str(tmp_path / "scenes"),
        "gaussian_model": str(tmp_path / "scenes" / scene_id / "3dgs_raw.ply"),
        "label_path": str(label_path),
        "actor_plan_path": str(actor_plan),
        "human_actor_ids": [actor_id],
        "peer_robot_ids": peer_robot_ids,
        "mission_families": ["dense_dynamic_humans"],
        "command": command,
        "env": {"GAUSSIAN_RENDER_BACKEND": "gsplat"},
        "robot_overlay_commands": robot_overlay_commands,
        "metadata": {"output_root": str(tmp_path / "out")},
    }


def test_work_items_extract_scene_actor_robot_resources(tmp_path: Path) -> None:
    estimates = ResourceEstimates(
        scene_vram_bytes=100,
        human_avatar_vram_bytes=10,
        robot_asset_vram_bytes=20,
        actor_plan_ram_bytes=5,
    )
    plan_payload = {
        "plans": [
            _plan(
                tmp_path,
                job_id="job_a",
                scene_id="scene_a",
                minimal_frames=11,
                with_robot=True,
            )
        ]
    }

    (item,) = build_work_items_from_render_plan(plan_payload, estimates=estimates)

    assert item.job_id == "job_a"
    assert item.frame_count_hint == 11
    assert item.estimated_vram_bytes == 130
    assert item.estimated_ram_bytes == 135
    assert {resource.kind for resource in item.resources} == {
        "actor_plan",
        "human_avatar",
        "robot_asset",
        "scene",
    }
    assert any(resource.kind == "actor_plan" and not resource.shareable_via_cuda_ipc for resource in item.resources)
    assert item.compatibility_key[-2:] == ("depth", "rgb_frames")


def test_scene_chunks_split_compatible_scene_work(tmp_path: Path) -> None:
    plan_payload = {
        "plans": [
            _plan(tmp_path, job_id="a0", scene_id="scene_a", actor_id="human_a"),
            _plan(tmp_path, job_id="a1", scene_id="scene_a", actor_id="human_b"),
            _plan(tmp_path, job_id="a2", scene_id="scene_a", actor_id="human_c"),
            _plan(tmp_path, job_id="b0", scene_id="scene_b", actor_id="human_d"),
        ]
    }
    items = build_work_items_from_render_plan(plan_payload)

    chunks = build_scene_chunks(items, max_items_per_chunk=2)

    assert [(chunk.scene_id, [item.job_id for item in chunk.items]) for chunk in chunks] == [
        ("scene_a", ["a0", "a1"]),
        ("scene_a", ["a2"]),
        ("scene_b", ["b0"]),
    ]
    assert chunks[0].frame_count_hint == 14
    assert any(resource.kind == "scene" for resource in chunks[0].resources)


def test_gpu_assignment_keeps_scene_chunks_together(tmp_path: Path) -> None:
    plan_payload = {
        "plans": [
            _plan(tmp_path, job_id="a0", scene_id="scene_a"),
            _plan(tmp_path, job_id="a1", scene_id="scene_a"),
            _plan(tmp_path, job_id="b0", scene_id="scene_b", actor_id="human_b"),
            _plan(tmp_path, job_id="c0", scene_id="scene_c", actor_id="human_c"),
        ]
    }
    chunks = build_scene_chunks(build_work_items_from_render_plan(plan_payload), max_items_per_chunk=1)

    assignments = assign_chunks_to_gpus(chunks, gpu_ids=["0", "1"])

    scene_a_gpus = {
        assignment.gpu_id
        for assignment in assignments
        for chunk in assignment.chunks
        if chunk.scene_id == "scene_a"
    }
    assert len(scene_a_gpus) == 1
    assert sum(len(assignment.chunks) for assignment in assignments) == len(chunks)


def test_gpu_assignment_supports_same_physical_gpu_worker_lanes(tmp_path: Path) -> None:
    plan_payload = {
        "plans": [
            _plan(tmp_path, job_id="a0", scene_id="scene_a"),
            _plan(tmp_path, job_id="b0", scene_id="scene_b", actor_id="human_b"),
            _plan(tmp_path, job_id="c0", scene_id="scene_c", actor_id="human_c"),
            _plan(tmp_path, job_id="d0", scene_id="scene_d", actor_id="human_d"),
        ]
    }
    chunks = build_scene_chunks(build_work_items_from_render_plan(plan_payload), max_items_per_chunk=1)

    assignments = assign_chunks_to_gpus(chunks, gpu_ids=["0", "0"])

    assert [assignment.gpu_id for assignment in assignments] == ["0", "0"]
    assert [assignment.assignment_id for assignment in assignments] == ["0_w00", "0_w01"]
    assert all(assignment.chunks for assignment in assignments)
    assert sum(len(assignment.chunks) for assignment in assignments) == len(chunks)


def test_persistent_schedule_json_has_assignments(tmp_path: Path) -> None:
    schedule = build_persistent_gpu_schedule(
        {"plans": [_plan(tmp_path, job_id="a0", scene_id="scene_a")]},
        gpu_ids=["0"],
        max_items_per_chunk=2,
    )

    payload = schedule.to_json_dict()

    assert payload["schema_version"] == "h100_persistent_schedule.v1"
    assert payload["work_item_count"] == 1
    assert payload["chunk_count"] == 1
    assert payload["assignments"][0]["assignment_id"] == "0"
    assert payload["assignments"][0]["gpu_id"] == "0"
    assert payload["assignments"][0]["chunks"][0]["job_ids"] == ["a0"]
    assert payload["includes_execution"] is False
    assert "plans" not in payload["assignments"][0]["chunks"][0]


def test_persistent_schedule_can_include_execution_payload(tmp_path: Path) -> None:
    schedule = build_persistent_gpu_schedule(
        {"plans": [_plan(tmp_path, job_id="a0", scene_id="scene_a")]},
        gpu_ids=["0"],
        max_items_per_chunk=2,
    )

    payload = schedule.to_json_dict(include_execution=True)
    chunk = payload["assignments"][0]["chunks"][0]

    assert payload["includes_execution"] is True
    assert chunk["plans"][0]["job_id"] == "a0"
    assert chunk["work_items"][0]["job_id"] == "a0"
    assert "--scene" in chunk["work_items"][0]["command"]
    assert chunk["work_items"][0]["env"] == {"GAUSSIAN_RENDER_BACKEND": "gsplat"}


def test_resource_cache_refuses_to_evict_leased_resources() -> None:
    cache = ResourceCacheState(capacity_vram_bytes=15)
    resource_a = ResourceRef("scene", "a", estimated_vram_bytes=10)
    resource_b = ResourceRef("scene", "b", estimated_vram_bytes=10)
    resource_c = ResourceRef("scene", "c", estimated_vram_bytes=10)

    assert [op.op for op in cache.acquire([resource_a])] == ["load", "acquire"]
    assert [op.op for op in cache.release([resource_a])] == ["release"]
    assert [op.op for op in cache.acquire([resource_b])] == ["evict", "load", "acquire"]
    with pytest.raises(RuntimeError, match="all eviction candidates are leased"):
        cache.acquire([resource_c])
    cache.release([resource_b])

    assert [op.op for op in cache.acquire([resource_c])] == ["evict", "load", "acquire"]
