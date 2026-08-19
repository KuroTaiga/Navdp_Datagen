from __future__ import annotations

import argparse

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


def test_group_entry_tasks_chunks_manifest_groups() -> None:
    grouped = {
        ("family", "source", "scene_a"): [{"id": index} for index in range(5)],
        ("family", "source", "scene_b"): [{"id": index} for index in range(2)],
    }

    tasks = bench._group_entry_tasks(grouped, max_manifests_per_task=2)

    assert [(key[2], [entry["id"] for entry in entries], chunk, count) for key, entries, _, chunk, count in tasks] == [
        ("scene_a", [0, 1], 0, 3),
        ("scene_a", [2, 3], 1, 3),
        ("scene_a", [4], 2, 3),
        ("scene_b", [0, 1], 0, 1),
    ]


def test_scene_entry_tasks_group_by_source_scene() -> None:
    entries = [
        {"family": "family_b", "source": "source", "scene": "scene_2", "id": "b"},
        {"family": "family_a", "source": "source", "scene": "scene_1", "id": "a0"},
        {"family": "family_c", "source": "source", "scene": "scene_1", "id": "a1"},
    ]
    grouped: dict[bench.GroupKey, list[dict[str, str]]] = {}
    for entry in entries:
        grouped.setdefault(bench._scene_entry_key(entry), []).append(entry)

    tasks = bench._group_entry_tasks(grouped, max_manifests_per_task=0)

    assert [(key, [entry["id"] for entry in task_entries]) for key, task_entries, *_ in tasks] == [
        (("source", "scene_1"), ["a0", "a1"]),
        (("source", "scene_2"), ["b"]),
    ]


def test_scene_order_task_groups_keep_scene_barriers_with_chunks() -> None:
    grouped = {
        ("source", "scene_b"): [{"id": "b0"}, {"id": "b1"}],
        ("source", "scene_a"): [{"id": "a0"}, {"id": "a1"}, {"id": "a2"}],
    }

    scene_groups = bench._scene_order_task_groups(grouped, max_manifests_per_task=2)

    assert [
        [(key, [entry["id"] for entry in entries], index, chunk, count) for key, entries, index, chunk, count in scene_tasks]
        for scene_tasks in scene_groups
    ] == [
        [
            (("source", "scene_a"), ["a0", "a1"], 0, 0, 2),
            (("source", "scene_a"), ["a2"], 1, 1, 2),
        ],
        [
            (("source", "scene_b"), ["b0", "b1"], 2, 0, 1),
        ],
    ]


def test_passes_filters_supports_scene_filter() -> None:
    entry = {"family": "family_a", "source": "source_a", "scene": "scene_a"}

    assert bench._passes_filters(entry, ["family_a"], ["source_a"], ["scene_a"]) is True
    assert bench._passes_filters(entry, ["family_a"], ["source_a"], ["scene_b"]) is False


def test_preemptible_task_commit_and_resume_marker(tmp_path) -> None:
    work_root = tmp_path / "scene.tmp.123"
    final_root = tmp_path / "scene"
    video = work_root / "renders" / "scene" / "label.mp4"
    metric = work_root / "metrics" / "group_0000.json"
    video.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    video.write_bytes(b"mp4")
    metric.write_text("{}", encoding="utf-8")
    record = {
        "status": "success",
        "output_root": str(work_root),
        "final_output_root": str(final_root),
        "videos": [str(video)],
        "metrics": [{"_path": str(metric), "frames_total": 12}],
    }

    committed = bench._commit_task_output_root(
        work_root=work_root,
        final_root=final_root,
        record=record,
        preemptible=True,
    )
    resumed = bench._load_done_record(committed)

    assert committed == final_root
    assert not work_root.exists()
    assert (final_root / "TASK_DONE.json").is_file()
    assert resumed is not None
    assert resumed["skipped_existing"] is True
    assert resumed["output_root"] == str(final_root)
    assert resumed["videos"] == [str(final_root / "renders" / "scene" / "label.mp4")]
    assert resumed["metrics"][0]["_path"] == str(final_root / "metrics" / "group_0000.json")


def test_render_entries_group_returns_existing_done_record(tmp_path) -> None:
    final_root = tmp_path / "scene"
    final_root.mkdir()
    record = {
        "status": "success",
        "output_root": str(final_root),
        "final_output_root": str(final_root),
        "source": "source",
        "scene": "scene",
        "videos": ["video.mp4"],
        "metrics": [],
    }
    bench._write_json(final_root / "TASK_DONE.json", {"record": record})
    args = argparse.Namespace(preemptible_output=True, resume=True)

    resumed = bench._render_entries_group(
        args,
        [],
        0,
        output_root=final_root,
        log_stem=tmp_path / "log",
        record_type="scene_ordered",
        family=None,
        source="source",
        scene="scene",
    )

    assert resumed["status"] == "success"
    assert resumed["skipped_existing"] is True
    assert resumed["output_root"] == str(final_root)


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
