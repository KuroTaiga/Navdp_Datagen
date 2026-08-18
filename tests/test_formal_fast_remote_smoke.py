from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.massgen import formal_fast_remote_smoke as smoke


def test_safe_component_handles_colon_family() -> None:
    assert (
        smoke._safe_component("navigate_with_social_constraints:queue_order")
        == "navigate_with_social_constraints__queue_order"
    )


def test_download_destination_groups_by_family_source_scene(tmp_path: Path) -> None:
    dest = smoke._download_destination(
        tmp_path,
        family="navigate_with_social_constraints:pedestrian_yield",
        source="InteriorGS",
        scene="0082_839951",
        remote_path="/remote/example.json",
    )

    assert dest == (
        tmp_path
        / "examples"
        / "navigate_with_social_constraints__pedestrian_yield"
        / "InteriorGS"
        / "0082_839951"
        / "example.json"
    )


def test_build_manifest_reports_structure_and_download_failures(tmp_path: Path) -> None:
    args = argparse.Namespace(
        host="pathGen_lxh",
        remote_root="/remote/root",
        output_root=tmp_path,
        seed=20260817,
        sample_mode="complete",
        expected_missions=500,
        examples_per_combo=1,
        download_mode="scene-jsons",
        no_download=False,
        max_remote_elapsed_sec=10.0,
    )
    summary = {
        "ok": False,
        "remote_elapsed_sec": 8.0,
        "family_count": 9,
        "sample_count": 36,
        "failure_count": 1,
    }
    downloads = [{"ok": False, "role": "mission_json"}]

    manifest = smoke._build_manifest(
        args=args,
        summary=summary,
        downloads=downloads,
        total_elapsed_sec=9.0,
    )

    assert manifest["structure_ok"] is False
    assert manifest["elapsed_ok"] is True
    assert manifest["download_ok"] is False
    assert manifest["ok"] is False
    assert manifest["download_failure_count"] == 1


def test_tar_members_include_json_dir_and_scene_reports() -> None:
    sample = {
        "scene_dir": "/remote/family/source/scene",
        "json_rel_dir": "navigate_with_social_constraints/queue_order_L4/jsons",
        "scene_metadata_files": [
            "/remote/family/source/scene/mass_example_manifest.json",
            "/remote/family/source/scene/mass_generation_report.json",
            "/outside/not_in_scene.json",
        ],
    }

    assert smoke._tar_members(sample) == [
        "navigate_with_social_constraints/queue_order_L4/jsons",
        "mass_example_manifest.json",
        "mass_generation_report.json",
    ]


def test_download_examples_copies_mission_and_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, stdout, stderr, text, check):
        calls.append(cmd)
        Path(cmd[-1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    summary = {
        "samples": [
            {
                "family": "deliver_to_human",
                "source": "CHINGMU_rescaled_1",
                "scene": "0005_858837",
                "examples": [
                    {
                        "mission_json": "/remote/mission.json",
                        "metadata_json": "/remote/mission_cornercase_metadata.json",
                    }
                ],
            }
        ]
    }

    downloads = smoke._download_examples(
        host="pathGen_lxh",
        output_root=tmp_path,
        summary=summary,
    )

    assert len(downloads) == 2
    assert all(item["ok"] for item in downloads)
    assert calls[0][:2] == ["scp", "pathGen_lxh:/remote/mission.json"]
    assert calls[1][:2] == [
        "scp",
        "pathGen_lxh:/remote/mission_cornercase_metadata.json",
    ]
