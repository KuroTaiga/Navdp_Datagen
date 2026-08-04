from __future__ import annotations

import json
from pathlib import Path


ACTION_INPUTS_PATH = Path("configs/massgen/avatar_action_generation_inputs.json")


def _load_inputs() -> dict[str, object]:
    with ACTION_INPUTS_PATH.open() as handle:
        return json.load(handle)


def test_avatar_action_generation_inputs_are_internally_consistent() -> None:
    data = _load_inputs()

    avatars = data["avatar_identities"]
    actions = data["action_inputs"]
    plans = data["avatar_action_plan"]

    assert data["defaults"]["candidate_generators"] == ["kimodo", "stmc"]
    assert data["defaults"]["output_root"] == "assets/human_actions/generated"
    assert data["request_expansion"]["priority_sources"] == ["required", "nice_to_have"]
    assert set(plans) == set(avatars)

    planned_pairs = set()
    for human_resource_id, plan in plans.items():
        assert human_resource_id in avatars
        for key in ("required_actions", "nice_to_have_actions"):
            assert plan[key]
            for action_id in plan[key]:
                assert action_id in actions
                planned_pairs.add((human_resource_id, action_id))

    request_ids = set()
    request_pairs = set()
    for request in data["generation_requests"]:
        human_resource_id = request["human_resource_id"]
        action_id = request["action_id"]
        action = actions[action_id]
        avatar = avatars[human_resource_id]

        assert request["request_id"] not in request_ids
        request_ids.add(request["request_id"])
        request_pairs.add((human_resource_id, action_id))

        assert request["appearance_id"] == avatar["appearance_id"]
        assert request["render_action_id"] == action["render_action_id"]
        assert request["generator"] == action["preferred_generator"]
        assert request["input_style"] == action["input_style"]
        assert request["duration_s"] == action["duration_s"]
        assert request["loop"] == action["loop"]
        assert request["root_motion_mode"] == action["root_motion_mode"]
        assert request["instruction"] == action["instruction"]
        assert request["quality_checks"] == action["quality_checks"]
        assert request["prompt"] == f"{avatar['appearance_prompt']}. {action['instruction']}"

        if request["input_style"] == "text_with_keypoints":
            assert request["keypoints"]
        else:
            assert request["input_style"] == "text"
            assert request["keypoints"] is None

        output_contract = request["output_contract"]
        base = f"assets/human_actions/generated/{human_resource_id}/{action_id}"
        assert output_contract["manifest_path"] == f"{base}/manifest.json"
        assert output_contract["ply_frame_dir"] == f"{base}/ply_frames"
        assert output_contract["smplx_frame_dir"] == f"{base}/smplx_frames"

    assert request_pairs == planned_pairs
