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
