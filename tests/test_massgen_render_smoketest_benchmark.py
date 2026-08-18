from __future__ import annotations

import pytest

from scripts.massgen import run_render_smoketest_benchmark as bench


def test_render_overhead_summary_uses_lifecycle_metrics() -> None:
    metrics = [
        {
            "frames_total": 80,
            "paths_total": 4,
            "paths_ok": 4,
            "duration_total_sec": 8.0,
            "lifecycle_seconds": {
                "render_loop_sec": 8.0,
                "process_total_sec": 20.0,
                "process_minus_render_loop_sec": 12.0,
            },
        },
        {
            "frames_total": 20,
            "paths_total": 1,
            "paths_ok": 1,
            "duration_total_sec": 2.0,
            "lifecycle_seconds": {
                "render_loop_sec": 2.0,
                "process_total_sec": 5.0,
                "process_minus_render_loop_sec": 3.0,
            },
        },
    ]

    summary = bench._render_overhead_summary(metrics, outer_render_elapsed_sec=55.0)

    assert summary["renderer_process_count"] == 2
    assert summary["nested_paths_total"] == 5
    assert summary["nested_paths_ok"] == 5
    assert summary["nested_frames_total"] == 100
    assert summary["nested_render_loop_sec"] == pytest.approx(10.0)
    assert summary["nested_process_total_sec"] == pytest.approx(25.0)
    assert summary["nested_setup_load_process_sec"] == pytest.approx(15.0)
    assert summary["hidden_overhead_sec"] == pytest.approx(45.0)
    assert summary["hidden_overhead_pct"] == pytest.approx(81.8181818)
    assert summary["process_wrapper_overhead_sec"] == pytest.approx(30.0)
    assert summary["process_launches_per_1000_frames"] == pytest.approx(20.0)
    assert summary["setup_seconds_per_frame"] == pytest.approx(0.45)
    assert summary["renderer_setup_seconds_per_frame"] == pytest.approx(0.15)


def test_render_overhead_summary_falls_back_for_old_metrics() -> None:
    summary = bench._render_overhead_summary(
        [
            {
                "frames_total": 50,
                "paths_total": 1,
                "paths_ok": 1,
                "duration_total_sec": 10.0,
            }
        ],
        outer_render_elapsed_sec=14.0,
    )

    assert summary["nested_render_loop_sec"] == pytest.approx(10.0)
    assert summary["nested_process_total_sec"] is None
    assert summary["process_wrapper_overhead_sec"] is None
    assert summary["hidden_overhead_sec"] == pytest.approx(4.0)


def test_run_level_render_overhead_summary_aggregates_success_records() -> None:
    run_summary = bench._summarize_render_overhead(
        [
            {
                "status": "success",
                "render_overhead": {
                    "outer_render_elapsed_sec": 12.0,
                    "renderer_process_count": 1,
                    "nested_paths_total": 2,
                    "nested_frames_total": 40,
                    "nested_duration_total_sec": 4.0,
                    "nested_render_loop_sec": 4.0,
                    "nested_process_total_sec": 8.0,
                    "nested_setup_load_process_sec": 4.0,
                },
            },
            {
                "status": "failed",
                "render_overhead": {
                    "outer_render_elapsed_sec": 100.0,
                    "renderer_process_count": 99,
                    "nested_frames_total": 1000,
                    "nested_render_loop_sec": 50.0,
                },
            },
            {
                "status": "success",
                "render_overhead": {
                    "outer_render_elapsed_sec": 18.0,
                    "renderer_process_count": 2,
                    "nested_paths_total": 3,
                    "nested_frames_total": 60,
                    "nested_duration_total_sec": 6.0,
                    "nested_render_loop_sec": 6.0,
                    "nested_process_total_sec": 10.0,
                    "nested_setup_load_process_sec": 4.0,
                },
            },
        ]
    )

    assert run_summary["record_count"] == 2
    assert run_summary["renderer_process_count"] == 3
    assert run_summary["nested_paths_total"] == 5
    assert run_summary["nested_frames_total"] == 100
    assert run_summary["nested_render_loop_sec"] == pytest.approx(10.0)
    assert run_summary["nested_process_total_sec"] == pytest.approx(18.0)
    assert run_summary["hidden_overhead_sec"] == pytest.approx(20.0)
    assert run_summary["process_wrapper_overhead_sec"] == pytest.approx(12.0)
    assert run_summary["process_launches_per_1000_frames"] == pytest.approx(30.0)


def test_grouped_render_command_combines_labels_and_actor_plan_mappings(tmp_path) -> None:
    base = [
        "python",
        "render_label_paths_telesim.py",
        "--scene",
        "scene_a",
        "--tasks-dir",
        str(tmp_path / "tasks"),
        "--output-dir",
        str(tmp_path / "renders"),
        "--video-backend",
        "cpu",
    ]
    plans = [
        {
            "command": [
                *base,
                "--label-id",
                "job_a",
                "--actor-plan-json",
                str(tmp_path / "actors_a.json"),
                "--metrics-json",
                str(tmp_path / "a.json"),
            ],
            "blockers": [],
            "robot_overlay_commands": [],
        },
        {
            "command": [
                *base,
                "--label-id",
                "job_b",
                "--actor-plan-json",
                str(tmp_path / "actors_b.json"),
                "--metrics-json",
                str(tmp_path / "b.json"),
            ],
            "blockers": [],
            "robot_overlay_commands": [],
        },
    ]

    grouped = bench._build_grouped_render_commands(plans, metrics_root=tmp_path / "metrics")

    assert len(grouped) == 1
    command = grouped[0]
    assert command.count("--label-id") == 2
    assert bench._option_values(command, "--label-id") == ["job_a", "job_b"]
    assert "--actor-plan-json" not in command
    assert bench._option_values(command, "--label-actor-plan-json") == [
        f"job_a={tmp_path / 'actors_a.json'}",
        f"job_b={tmp_path / 'actors_b.json'}",
    ]
    assert bench._single_option_value(command, "--metrics-json") == str(tmp_path / "metrics" / "group_0000.json")


def test_grouped_render_command_chunks_compatible_labels(tmp_path) -> None:
    base = [
        "python",
        "render_label_paths_telesim.py",
        "--scene",
        "scene_a",
        "--tasks-dir",
        str(tmp_path / "tasks"),
        "--output-dir",
        str(tmp_path / "renders"),
        "--video-backend",
        "cpu",
    ]
    plans = [
        {
            "command": [
                *base,
                "--label-id",
                f"job_{index}",
                "--actor-plan-json",
                str(tmp_path / f"actors_{index}.json"),
                "--metrics-json",
                str(tmp_path / f"{index}.json"),
            ],
            "blockers": [],
            "robot_overlay_commands": [],
        }
        for index in range(5)
    ]

    grouped = bench._build_grouped_render_commands(
        plans,
        metrics_root=tmp_path / "metrics",
        max_labels_per_command=2,
    )

    assert len(grouped) == 3
    assert [bench._option_values(command, "--label-id") for command in grouped] == [
        ["job_0", "job_1"],
        ["job_2", "job_3"],
        ["job_4"],
    ]
    assert [bench._single_option_value(command, "--metrics-json") for command in grouped] == [
        str(tmp_path / "metrics" / "group_0000.json"),
        str(tmp_path / "metrics" / "group_0001.json"),
        str(tmp_path / "metrics" / "group_0002.json"),
    ]


def test_grouped_render_command_keeps_incompatible_render_options_separate(tmp_path) -> None:
    plans = [
        {
            "command": [
                "python",
                "render_label_paths_telesim.py",
                "--scene",
                "scene_a",
                "--resolution",
                "320",
                "240",
                "--label-id",
                "job_a",
                "--metrics-json",
                str(tmp_path / "a.json"),
            ],
            "blockers": [],
            "robot_overlay_commands": [],
        },
        {
            "command": [
                "python",
                "render_label_paths_telesim.py",
                "--scene",
                "scene_a",
                "--resolution",
                "640",
                "480",
                "--label-id",
                "job_b",
                "--metrics-json",
                str(tmp_path / "b.json"),
            ],
            "blockers": [],
            "robot_overlay_commands": [],
        },
    ]

    grouped = bench._build_grouped_render_commands(plans, metrics_root=tmp_path / "metrics")

    assert len(grouped) == 2
