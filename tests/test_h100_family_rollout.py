from __future__ import annotations

from pathlib import Path

from scripts.massgen import run_family_rollout_h100 as h100


def test_parse_gpu_devices_from_comma_list() -> None:
    assert h100._parse_gpu_devices("0,1, 3") == ["0", "1", "3"]
    assert h100._parse_gpu_devices("") == []
    assert h100._parse_gpu_devices(None) == []


def test_build_worker_slots_spreads_lanes_across_gpus() -> None:
    slots = h100._build_worker_slots(
        ["0", "1"],
        jobs_per_gpu=2,
        cpu_cores=16,
    )

    assert [slot.gpu_id for slot in slots] == ["0", "1", "0", "1"]
    assert [slot.cpu_threads for slot in slots] == [4, 4, 4, 4]
    assert [slot.core_list for slot in slots] == ["0,1,2,3", "4,5,6,7", "8,9,10,11", "12,13,14,15"]


def test_build_worker_slots_respects_max_workers_and_cpu_override() -> None:
    slots = h100._build_worker_slots(
        ["0", "1", "2", "3"],
        jobs_per_gpu=4,
        cpu_cores=120,
        cpu_cores_per_worker=6,
        max_workers=5,
    )

    assert len(slots) == 5
    assert [slot.gpu_id for slot in slots] == ["0", "1", "2", "3", "0"]
    assert all(slot.cpu_threads == 6 for slot in slots)


def test_worker_env_pins_gpu_and_thread_budget(tmp_path: Path) -> None:
    slot = h100.WorkerSlot(slot_id=2, gpu_id="3", cpu_cores=(12, 13, 14), cpu_threads=3)
    env_root = tmp_path / "env"
    (env_root / "bin").mkdir(parents=True)
    (env_root / "lib").mkdir()

    env = h100._worker_env(
        base_env={"PATH": "/bin"},
        slot=slot,
        ffmpeg_bin=tmp_path / "ffmpeg",
        python_bin=env_root / "bin" / "python",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["TORCH_NUM_THREADS"] == "3"
    assert env["GAUSSIAN_RENDER_BACKEND"] == "gsplat"
    assert env["PYOPENGL_PLATFORM"] == "egl"
    assert env["LD_LIBRARY_PATH"] == str(env_root / "lib")
    assert env["IMAGEIO_FFMPEG_EXE"] == str(tmp_path / "ffmpeg")


def test_persistent_runner_env_forwards_ffmpeg_override(tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    (env_root / "bin").mkdir(parents=True)
    (env_root / "lib").mkdir()
    args = h100._parse_args(
        [
            "--package-root",
            str(tmp_path / "package"),
            "--results-root",
            str(tmp_path / "results"),
            "--python-bin",
            str(env_root / "bin" / "python"),
            "--ffmpeg-bin",
            str(tmp_path / "bin" / "ffmpeg"),
        ]
    )

    env = h100._persistent_runner_env(args)

    assert env["GAUSSIAN_RENDER_BACKEND"] == "gsplat"
    assert env["PYOPENGL_PLATFORM"] == "egl"
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(env_root / "lib")
    assert env["IMAGEIO_FFMPEG_EXE"] == str(tmp_path / "bin" / "ffmpeg")
    assert env["FFMPEG_BIN"] == str(tmp_path / "bin" / "ffmpeg")


def test_slot_gpu_ids_preserve_repeated_worker_lanes() -> None:
    slots = h100._build_worker_slots(
        ["0", "1"],
        jobs_per_gpu=3,
        cpu_cores=24,
    )

    assert h100._slot_gpu_ids(slots) == ["0", "1", "0", "1", "0", "1"]


def test_assignment_cpu_core_map_follows_schedule_order(tmp_path: Path) -> None:
    schedule_json = tmp_path / "persistent_schedule.json"
    schedule_json.write_text(
        """
        {
          "schema_version": "h100_persistent_schedule.v1",
          "assignments": [
            {"assignment_id": "0_w00", "gpu_id": "0", "chunks": []},
            {"assignment_id": "0_w01", "gpu_id": "0", "chunks": []}
          ]
        }
        """,
        encoding="utf-8",
    )
    slots = [
        h100.WorkerSlot(slot_id=0, gpu_id="0", cpu_cores=(0, 1, 2), cpu_threads=3),
        h100.WorkerSlot(slot_id=1, gpu_id="0", cpu_cores=(3, 4, 5), cpu_threads=3),
    ]

    assert h100._assignment_cpu_core_map(schedule_json, slots) == {
        "0_w00": [0, 1, 2],
        "0_w01": [3, 4, 5],
    }


def test_persistent_plan_command_uses_one_lane_per_slot(tmp_path: Path) -> None:
    args = h100._parse_args(
        [
            "--package-root",
            str(tmp_path / "package"),
            "--results-root",
            str(tmp_path / "results"),
            "--gpu-devices",
            "0",
            "--jobs-per-gpu",
            "2",
            "--renders-per-family-source-scene",
            "50",
        ]
    )
    args.package_root = args.package_root.resolve()
    args.results_root = args.results_root.resolve()
    args.python_bin = args.python_bin.resolve()
    args.render_script = args.render_script.resolve()
    slots = [
        h100.WorkerSlot(slot_id=0, gpu_id="0", cpu_cores=(0,), cpu_threads=1),
        h100.WorkerSlot(slot_id=1, gpu_id="0", cpu_cores=(1,), cpu_threads=1),
    ]

    command = h100._persistent_plan_command(args, slots=slots)

    assert command.count("--gpu-id") == 2
    assert command[command.index("--workers-per-gpu") + 1] == "1"
    assert "--renders-per-family-source-scene" in command


def test_copy_package_metadata_copies_generic_jsons(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "family_index.json").write_text("{}", encoding="utf-8")
    (package / "action_catalog_h100_avatar.json").write_text("{}", encoding="utf-8")
    (package / "not_json.txt").write_text("skip", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()

    h100._copy_package_metadata(package, results)

    assert (results / "family_index.json").is_file()
    assert (results / "action_catalog_h100_avatar.json").is_file()
    assert not (results / "not_json.txt").exists()


def test_gpu_utilization_summary_reports_soft_target() -> None:
    summary = h100._summarize_gpu_utilization(
        [
            {"gpu_index": "0", "gpu_util_pct": 70.0},
            {"gpu_index": "0", "gpu_util_pct": 90.0},
            {"gpu_index": "1", "gpu_util_pct": 60.0},
        ],
        target_gpu_util=80.0,
    )

    assert summary["per_gpu"]["0"]["avg_gpu_util_pct"] == 80.0
    assert summary["per_gpu"]["0"]["target_met"] is True
    assert summary["per_gpu"]["1"]["target_met"] is False
    assert summary["target_met"] is False
