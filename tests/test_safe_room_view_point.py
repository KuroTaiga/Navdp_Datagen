from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.safe_room_view_point import choose_safe_room_view_point


ROOM_POLYGON = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]


def _write_scene(tmp_path: Path, occupancy: np.ndarray, labels=None, structure=None) -> Path:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    Image.fromarray(occupancy.astype(np.uint8), mode="L").save(scene_dir / "occupancy.png")
    h, w = occupancy.shape
    scale = 0.1
    left = -1.0
    top = 1.0
    (scene_dir / "occupancy.json").write_text(
        json.dumps(
            {
                "scale": scale,
                "min": [left, top - h * scale, 0.0],
                "max": [left + w * scale, top, 1.0],
                "lower": [left, top - h * scale, 0.0],
                "upper": [left + w * scale, top, 1.0],
            }
        ),
        encoding="utf-8",
    )
    if labels is not None:
        (scene_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    if structure is not None:
        (scene_dir / "structure.json").write_text(json.dumps(structure), encoding="utf-8")
    return scene_dir


def test_original_valid_point_is_kept(tmp_path: Path) -> None:
    scene_dir = _write_scene(tmp_path, np.full((21, 21), 255, dtype=np.uint8), labels=[], structure={})

    result = choose_safe_room_view_point(scene_dir, (0.0, 0.0), room_polygon=ROOM_POLYGON, occupancy_clearance_m=0.0)

    assert result.status == "original_valid"
    assert result.selected_xy == (0.0, 0.0)
    assert result.manual_verification_required is False


def test_point_inside_object_moves_to_nearest_free_point(tmp_path: Path) -> None:
    labels = [
        {
            "ins_id": "chair_12",
            "label": "chair",
            "bounding_box": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 1.0},
            ],
        }
    ]
    scene_dir = _write_scene(tmp_path, np.full((21, 21), 255, dtype=np.uint8), labels=labels, structure={})

    result = choose_safe_room_view_point(scene_dir, (0.0, 0.0), room_polygon=ROOM_POLYGON, occupancy_clearance_m=0.0)

    assert result.status == "adjusted_within_0.5m"
    assert result.selected_xy != (0.0, 0.0)
    assert result.manual_verification_required is False
    assert result.collided_label_ids == ["chair_12"]


def test_point_above_low_object_is_kept(tmp_path: Path) -> None:
    occupancy = np.full((21, 21), 255, dtype=np.uint8)
    occupancy[10, 10] = 0
    labels = [
        {
            "ins_id": "desk_1",
            "label": "desk",
            "bounding_box": [
                {"x": -0.2, "y": -0.2, "z": 0.0},
                {"x": 0.2, "y": 0.2, "z": 0.8},
            ],
        }
    ]
    scene_dir = _write_scene(tmp_path, occupancy, labels=labels, structure={})

    result = choose_safe_room_view_point(
        scene_dir,
        (0.0, 0.0),
        room_polygon=ROOM_POLYGON,
        camera_z=1.5,
        object_vertical_clearance_m=0.2,
        occupancy_clearance_m=0.0,
    )

    assert result.status == "original_valid"
    assert result.selected_xy == (0.0, 0.0)
    assert result.manual_verification_required is False


def test_point_inside_tall_object_still_moves(tmp_path: Path) -> None:
    labels = [
        {
            "ins_id": "cabinet_1",
            "label": "cabinet",
            "bounding_box": [
                {"x": -0.2, "y": -0.2, "z": 0.0},
                {"x": 0.2, "y": 0.2, "z": 1.6},
            ],
        }
    ]
    scene_dir = _write_scene(tmp_path, np.full((21, 21), 255, dtype=np.uint8), labels=labels, structure={})

    result = choose_safe_room_view_point(
        scene_dir,
        (0.0, 0.0),
        room_polygon=ROOM_POLYGON,
        camera_z=1.5,
        occupancy_clearance_m=0.0,
    )

    assert result.status == "adjusted_within_0.5m"
    assert result.selected_xy != (0.0, 0.0)
    assert result.collided_label_ids == ["cabinet_1"]


def test_point_on_black_occupancy_moves_to_free_point(tmp_path: Path) -> None:
    occupancy = np.full((21, 21), 255, dtype=np.uint8)
    occupancy[10, 10] = 0
    scene_dir = _write_scene(tmp_path, occupancy, labels=[], structure={})

    result = choose_safe_room_view_point(scene_dir, (0.0, 0.0), room_polygon=ROOM_POLYGON, occupancy_clearance_m=0.0)

    assert result.status == "adjusted_within_0.5m"
    assert result.selected_xy != (0.0, 0.0)
    assert "original_non_free_occupancy" in result.reasons


def test_point_uses_second_search_radius_when_needed(tmp_path: Path) -> None:
    labels = [
        {
            "ins_id": "wide_table",
            "label": "table",
            "bounding_box": [
                {"x": -0.5, "y": -0.5, "z": 0.0},
                {"x": 0.5, "y": 0.5, "z": 1.0},
            ],
        }
    ]
    scene_dir = _write_scene(tmp_path, np.full((21, 21), 255, dtype=np.uint8), labels=labels, structure={})

    result = choose_safe_room_view_point(scene_dir, (0.0, 0.0), room_polygon=ROOM_POLYGON, occupancy_clearance_m=0.0)

    assert result.status == "adjusted_within_1.0m"
    assert result.search_radius_m == 1.0
    assert result.manual_verification_required is False


def test_point_on_wall_or_door_moves(tmp_path: Path) -> None:
    structure = {
        "walls": [{"id": "wall_a", "location": [[0.0, -0.5], [0.0, 0.5]]}],
        "holes": [{"type": "DOOR", "profile": [[0.4, -0.2, 0.0], [0.4, 0.2, 0.0]]}],
    }
    scene_dir = _write_scene(tmp_path, np.full((21, 21), 255, dtype=np.uint8), labels=[], structure=structure)

    result = choose_safe_room_view_point(
        scene_dir,
        (0.0, 0.0),
        room_polygon=ROOM_POLYGON,
        occupancy_clearance_m=0.0,
        structure_margin_m=0.03,
    )

    assert result.status == "adjusted_within_0.5m"
    assert result.selected_xy != (0.0, 0.0)
    assert "original_overlaps_structure" in result.reasons
    assert result.collided_structure_ids == ["walls:wall_a:0"]


def test_unfixable_point_requires_manual_verification(tmp_path: Path) -> None:
    scene_dir = _write_scene(tmp_path, np.zeros((21, 21), dtype=np.uint8), labels=[], structure={})

    result = choose_safe_room_view_point(scene_dir, (0.0, 0.0), room_polygon=ROOM_POLYGON, occupancy_clearance_m=0.0)

    assert result.status == "manual_verification_required"
    assert result.selected_xy == (0.0, 0.0)
    assert result.manual_verification_required is True
