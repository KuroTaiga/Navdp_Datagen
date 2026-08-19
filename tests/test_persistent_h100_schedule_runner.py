from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.massgen import plan_persistent_h100_schedule as planner
from scripts.massgen import run_persistent_h100_schedule as runner
from scripts.massgen import run_render_smoketest_benchmark as smoke


def _chunk(tmp_path: Path, *, overlay: bool = False) -> dict[str, object]:
    command = [
        "python",
        "render_label_paths_telesim.py",
        "--scene",
        "scene_a",
        "--tasks-dir",
        str(tmp_path / "tasks"),
        "--output-dir",
        str(tmp_path / "old_renders"),
        "--label-id",
        "job_a",
        "--actor-plan-json",
        str(tmp_path / "actors.json"),
        "--metrics-json",
        str(tmp_path / "old_metrics.json"),
        "--video-backend",
        "cpu",
    ]
    return {
        "chunk_id": "scene_a_g0000_c0000",
        "scene_id": "scene_a",
        "job_ids": ["job_a"],
        "frame_count_hint": 32,
        "estimated_vram_bytes": 123,
        "estimated_ram_bytes": 456,
        "plans": [
            {
                "status": "ready",
                "job_id": "job_a",
                "scene_id": "scene_a",
                "command": command,
                "env": {"GAUSSIAN_RENDER_BACKEND": "gsplat"},
                "blockers": [],
                "robot_overlay_commands": [{"command": ["robot_overlay"]}] if overlay else [],
            }
        ],
    }


def test_rewrite_plan_for_chunk_moves_outputs_under_chunk_root(tmp_path: Path) -> None:
    chunk = _chunk(tmp_path)
    output_root = tmp_path / "chunk_out"

    rewritten = runner._rewrite_plan_for_chunk(chunk["plans"][0], output_root=output_root)
    command = rewritten["command"]

    assert smoke._single_option_value(command, "--output-dir") == str(output_root / "renders")
    assert smoke._single_option_value(command, "--metrics-json") == str(
        output_root / "metrics" / "job_a.json"
    )


def test_render_chunk_dry_run_writes_preemptible_done_marker(tmp_path: Path) -> None:
    args = argparse.Namespace(
        results_root=tmp_path / "results",
        repo_root=tmp_path,
        group_max_labels_per_command=0,
        preemptible_output=True,
        resume=True,
        dry_run=True,
    )

    record = runner._render_chunk(
        args,
        gpu_id="0",
        chunk=_chunk(tmp_path),
        assignment_index=0,
        chunk_index=0,
    )

    final_root = Path(record["final_output_root"])
    assert record["status"] == "success"
    assert final_root.name == "scene_a_g0000_c0000"
    assert (final_root / "TASK_DONE.json").is_file()
    marker = json.loads((final_root / "TASK_DONE.json").read_text(encoding="utf-8"))
    assert marker["record"]["chunk_id"] == "scene_a_g0000_c0000"


def test_render_chunk_blocks_robot_overlay_for_phase_a(tmp_path: Path) -> None:
    args = argparse.Namespace(
        results_root=tmp_path / "results",
        repo_root=tmp_path,
        group_max_labels_per_command=0,
        preemptible_output=False,
        resume=False,
        dry_run=False,
    )

    record = runner._render_chunk(
        args,
        gpu_id="0",
        chunk=_chunk(tmp_path, overlay=True),
        assignment_index=0,
        chunk_index=0,
    )

    assert record["status"] == "blocked"
    assert "robot overlay plans are not supported" in record["blockers"][0]


def test_metrics_success_requires_no_fatal_paths() -> None:
    assert (
        runner._metrics_indicate_success(
            [{"paths_ok": 1, "paths_fatal": 0, "paths_oom": 0}],
            expected_paths=1,
        )
        is True
    )
    assert (
        runner._metrics_indicate_success(
            [{"paths_ok": 1, "paths_fatal": 1, "paths_oom": 0}],
            expected_paths=2,
        )
        is False
    )
    assert (
        runner._metrics_indicate_success(
            [{"paths_ok": 1, "paths_fatal": 0, "paths_oom": 0}],
            expected_paths=2,
        )
        is False
    )


def _write_png_header(path: Path, *, width: int = 32, height: int = 32) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + int(width).to_bytes(4, "big")
        + int(height).to_bytes(4, "big")
        + b"\x08\x00\x00\x00\x00"
    )


def _write_package(tmp_path: Path) -> Path:
    scene_dir = tmp_path / "package" / "scenes" / "scene_a"
    scene_dir.mkdir(parents=True)
    (scene_dir / "occupancy.json").write_text(
        json.dumps(
            {
                "scale": 0.5,
                "min": [0.0, 0.0, 0.0],
                "max": [16.0, 16.0, 3.0],
                "lower": [0.0, 0.0, 0.0],
                "upper": [16.0, 16.0, 2.4],
            }
        ),
        encoding="utf-8",
    )
    _write_png_header(scene_dir / "occupancy.png")
    scene_ply = scene_dir / "3dgs_raw.ply"
    scene_ply.write_text("ply\n", encoding="utf-8")
    action_dir = tmp_path / "package" / "actions" / "stand"
    action_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "massgen_render_manifest.v1",
        "source": {"scenario_id": "scenario_a", "scene_id": "scene_a"},
        "scene_assets": {"splat_model_path": str(scene_ply)},
        "timing": {"fps": 10.0},
        "mission_families": ["deliver_to_human"],
        "sensor_rigs": {
            "rig_a": {
                "sensors": [
                    {
                        "name": "fpv_rgb",
                        "intrinsics": {"width": 320, "height": 240, "fov_y_deg": 70.0},
                        "clipping_range_m": [0.01, 30.0],
                    }
                ]
            }
        },
        "actors": {
            "humans": [
                {
                    "actor_id": "human_a",
                    "trajectory": [
                        {"time_s": 0.0, "position": [4.0, 4.0, 0.0], "yaw_rad": 0.0},
                        {"time_s": 1.0, "position": [4.0, 4.0, 0.0], "yaw_rad": 0.0},
                    ],
                    "action_segments": [
                        {
                            "render_action_id": "stand",
                            "action_sequence_id": "stand",
                            "asset": {
                                "ply_frame_dir": str(action_dir),
                                "root_motion_mode": "stationary",
                                "fps": 10.0,
                            },
                        }
                    ],
                }
            ]
        },
        "jobs": [
            {
                "job_id": "job_a",
                "scene_id": "scene_a",
                "viewpoint_robot_id": "robot_a",
                "mission_families": ["deliver_to_human"],
                "assigned_mission_ids": ["mission_a"],
                "human_actor_ids": ["human_a"],
                "sensors": [{"rig_id": "rig_a", "sensor_name": "fpv_rgb"}],
                "camera": {
                    "trajectory": [
                        {"time_s": 0.0, "position": [1.0, 1.0, 0.0]},
                        {"time_s": 1.0, "position": [2.0, 2.0, 0.0]},
                    ]
                },
            }
        ],
    }
    package_root = tmp_path / "package"
    manifest_path = package_root / "manifests" / "job_a.render_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index = {
        "entries": [
            {
                "family": "deliver_to_human",
                "source": "source_a",
                "scene": "scene_a",
                "render_manifest_json": str(manifest_path.relative_to(package_root)),
            }
        ]
    }
    (package_root / "smoketest_package_index.json").write_text(json.dumps(index), encoding="utf-8")
    return package_root


def test_planner_builds_executable_schedule_from_package(tmp_path: Path) -> None:
    package_root = _write_package(tmp_path)
    args = argparse.Namespace(
        package_root=package_root,
        materialized_root=tmp_path / "materialized",
        render_plan_output_json=None,
        output_json=tmp_path / "schedule.json",
        family=["deliver_to_human"],
        source=None,
        scene=None,
        max_renders=0,
        renders_per_family_source_scene=0,
        python_bin=sys.executable,
        render_script=Path("render_label_paths_telesim.py"),
        video_backend="cpu",
        device="cuda",
        minimal_frames=8,
        actor_gpu_resident=True,
        actor_runtime_cache=True,
    )

    plan_payload = planner._build_render_plan_from_package(args)
    schedule = planner.build_persistent_gpu_schedule(
        plan_payload,
        gpu_ids=["0"],
        max_items_per_chunk=2,
    ).to_json_dict(include_execution=True)

    chunk = schedule["assignments"][0]["chunks"][0]
    assert plan_payload["status"] == "ready"
    assert plan_payload["selected_entry_count"] == 1
    assert plan_payload["plans"][0]["metadata"]["output_root"] == str(tmp_path / "materialized")
    assert chunk["plans"][0]["job_id"] == "job_a"
    assert "--actor-runtime-cache" in chunk["plans"][0]["command"]
    assert Path(chunk["plans"][0]["label_path"]).is_file()
    assert Path(chunk["plans"][0]["actor_plan_path"]).is_file()
