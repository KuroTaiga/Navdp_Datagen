from __future__ import annotations

from pathlib import Path

from scripts.massgen.package_family_smoke_examples import (
    _candidate_scenario,
    _copy_or_generate_visual,
    _scenario_for_smoke,
    _scenario_overrides,
)


def _scenario() -> dict:
    return {
        "scene_id": "0011_859081",
        "scene_assets": {"dataset": "CHINGMU_rescaled_3"},
        "robots": [
            {
                "robot_id": "robot_alpha",
                "trajectory": [
                    {"t": float(index), "map_pose": {"x": float(index), "y": 0.0}}
                    for index in range(9)
                ],
            }
        ],
        "humans": [
            {
                "human_id": "human_target",
                "trajectory": [
                    {"t": float(index), "map_pose": {"x": 0.0, "y": float(index)}}
                    for index in range(9)
                ],
                "action_sequences": [
                    {
                        "action_id": "walk",
                        "ply_frame_dir": "/old/path",
                        "manifest_path": "/old/manifest.json",
                        "smplx_frame_dir": "/old/smplx",
                    }
                ],
            }
        ],
    }


def test_scenario_for_smoke_can_preserve_scene_and_full_trajectories() -> None:
    payload = _scenario_for_smoke(
        _scenario(),
        scene_id=None,
        scene_ply="/mnt/DATA/dongjk/navdp_data/CHINGMU_rescaled_3/0011_859081/3dgs_compressed.ply",
        actor_ply_frame_dir="/actors/human_a",
        thin_trajectories=False,
    )

    assert payload["scene_id"] == "0011_859081"
    assert payload["scene_assets"]["dataset"] == "CHINGMU_rescaled_3"
    assert (
        payload["scene_assets"]["splat_model_path"]
        == "/mnt/DATA/dongjk/navdp_data/CHINGMU_rescaled_3/0011_859081/3dgs_compressed.ply"
    )
    assert len(payload["robots"][0]["trajectory"]) == 9
    assert len(payload["humans"][0]["trajectory"]) == 9
    assert payload["humans"][0]["action_sequences"][0]["ply_frame_dir"] == "/actors/human_a"
    assert payload["humans"][0]["action_sequences"][0]["manifest_path"] is None
    assert payload["humans"][0]["action_sequences"][0]["pre_generated"] is True


def test_scenario_for_smoke_can_still_substitute_scene_and_thin() -> None:
    payload = _scenario_for_smoke(
        _scenario(),
        scene_id="0030_839913",
        scene_ply="/mnt/DATA/dongjk/navdp_data/scenes/0030_839913/3dgs_compressed.ply",
        actor_ply_frame_dir="/actors/human_a",
        thin_trajectories=True,
    )

    assert payload["scene_id"] == "0030_839913"
    assert (
        payload["scene_assets"]["splat_model_path"]
        == "/mnt/DATA/dongjk/navdp_data/scenes/0030_839913/3dgs_compressed.ply"
    )
    assert len(payload["robots"][0]["trajectory"]) == 6
    assert len(payload["humans"][0]["trajectory"]) == 6


def test_scenario_for_smoke_assigns_stable_identity_per_human() -> None:
    scenario = _scenario()
    scenario["humans"].append(
        {
            "human_id": "human_peer",
            "trajectory": [
                {"t": float(index), "map_pose": {"x": 2.0, "y": float(index)}}
                for index in range(9)
            ],
            "action_sequences": [
                {"action_id": "stand", "ply_frame_dir": "/old/a"},
                {"action_id": "wave", "ply_frame_dir": "/old/b"},
            ],
        }
    )

    payload = _scenario_for_smoke(
        scenario,
        scene_id=None,
        scene_ply=None,
        actor_ply_frame_dir="/actors/fallback",
        actor_ply_frame_dirs=["/actors/1018", "/actors/10395"],
        thin_trajectories=False,
    )

    first, second = payload["humans"]
    assert first["metadata"]["renderer_actor_identity_id"] == "1018"
    assert second["metadata"]["renderer_actor_identity_id"] == "10395"
    assert {seq["ply_frame_dir"] for seq in first["action_sequences"]} == {"/actors/1018"}
    assert {seq["ply_frame_dir"] for seq in second["action_sequences"]} == {"/actors/10395"}


def test_copy_or_generate_visual_requires_exact_scenario_match(tmp_path) -> None:
    visual_dir = tmp_path / "visual" / "dense_dynamic_humans" / "visualizations"
    visual_dir.mkdir(parents=True)
    stale_visual = visual_dir / "example_scene_v9944_bev_trajectory.png"
    stale_visual.write_bytes(b"stale")

    family_dir = tmp_path / "family"
    visual = _copy_or_generate_visual(
        visual_root=tmp_path / "visual",
        source_rel="dense_dynamic_humans",
        scenario=_scenario(),
        scenario_stem="example_scene_v10000",
        family_dir=family_dir,
        family_key="dense_dynamic_humans",
    )

    assert visual["primary"] == str(family_dir / "example_visualization.png")
    assert visual["source"] == "generated_from_scenario"
    assert visual["paths"] == [str(family_dir / "example_visualization.png")]
    assert not (family_dir / stale_visual.name).exists()
    assert (family_dir / "example_visualization.png").is_file()


def test_copy_or_generate_visual_copies_exact_match_sidecars(tmp_path) -> None:
    visual_dir = tmp_path / "visual" / "dense_dynamic_humans" / "visualizations"
    visual_dir.mkdir(parents=True)
    exact_png = visual_dir / "example_scene_v10000_bev_trajectory.png"
    exact_png.write_bytes(b"exact")
    exact_gif = visual_dir / "example_scene_v10000_bev_trajectory.gif"
    exact_gif.write_bytes(b"gif")

    family_dir = tmp_path / "family"
    visual = _copy_or_generate_visual(
        visual_root=tmp_path / "visual",
        source_rel="dense_dynamic_humans",
        scenario=_scenario(),
        scenario_stem="example_scene_v10000",
        family_dir=family_dir,
        family_key="dense_dynamic_humans",
    )

    assert visual["primary"] == str(family_dir / exact_png.name)
    assert visual["source"] == "planner_exact"
    assert set(visual["paths"]) == {
        str(family_dir / exact_png.name),
        str(family_dir / exact_gif.name),
    }
    assert (family_dir / exact_png.name).read_bytes() == b"exact"
    assert (family_dir / exact_gif.name).read_bytes() == b"gif"


def test_scenario_overrides_parse_family_paths() -> None:
    overrides = _scenario_overrides(["dense_dynamic_humans=/tmp/example.json"])

    assert overrides == {"dense_dynamic_humans": Path("/tmp/example.json")}


def test_candidate_scenario_prefers_planner_png_gif_pair(tmp_path) -> None:
    source_root = tmp_path / "source"
    json_dir = source_root / "deliver_to_human" / "jsons"
    json_dir.mkdir(parents=True)
    first = json_dir / "example_v10000.json"
    first.write_text("{}", encoding="utf-8")
    visualized = json_dir / "example_v9997.json"
    visualized.write_text("{}", encoding="utf-8")
    visual_dir = tmp_path / "visual" / "deliver_to_human" / "visualizations"
    visual_dir.mkdir(parents=True)
    (visual_dir / "example_v9997_bev_trajectory.png").write_bytes(b"png")
    (visual_dir / "example_v9997_bev_trajectory.gif").write_bytes(b"gif")

    selected = _candidate_scenario(
        source_root,
        "deliver_to_human",
        visual_root=tmp_path / "visual",
    )

    assert selected == visualized
