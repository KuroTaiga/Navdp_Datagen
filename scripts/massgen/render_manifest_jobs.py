#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.render_executor import (  # noqa: E402
    build_render_plans,
    execute_render_plans,
    format_plan_text,
    load_render_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute MassGen render jobs from a render manifest."
    )
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional prepare_render_run summary. If present and blocked, execution is refused.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root for generated label inputs, metrics, and renders. Defaults to <manifest-dir>/render_jobs.",
    )
    parser.add_argument("--scenes-dir", type=Path, default=None)
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--render-script", type=Path, default=REPO_ROOT / "render_label_paths_telesim.py")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--robot-id", action="append", default=None)
    parser.add_argument("--sensor", dest="sensor_names", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--write-inputs",
        action="store_true",
        help="Write renderer label-path JSONs from manifest robot trajectories.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the planned commands. Refuses jobs with blockers.",
    )
    parser.add_argument("--video-backend", default="nvenc", choices=["cpu", "nvenc", "gpu"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-depth-maps", action="store_true")
    parser.add_argument("--save-rgb-frames", action="store_true")
    parser.add_argument(
        "--minimal-frames",
        type=int,
        default=None,
        help="If >0, truncate renderer jobs to the first N frames for smoke/benchmark runs.",
    )
    parser.add_argument(
        "--actor-gpu-resident",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache human actor sequences on GPU for per-frame transforms.",
    )
    parser.add_argument(
        "--save-actor-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-frame actor candidate/visible/culled/rendered metadata.",
    )
    parser.add_argument(
        "--robot-overlay-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "render" / "assets" / "render_glb_robot_overlay.py",
    )
    parser.add_argument(
        "--robot-glb",
        type=Path,
        default=REPO_ROOT / "assets" / "robots" / "g1_29dof_mode_16.glb",
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=REPO_ROOT / "data" / "g1_description" / "g1_29dof_mode_16.urdf",
    )
    parser.add_argument(
        "--kimodo-smplx-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "walking_kimodo",
        help="Kimodo SMPL-X frames used to retarget G1 AMO/joint poses for robot overlays.",
    )
    parser.add_argument("--robot-compose-mode", choices=["foreground", "depth"], default="depth")
    parser.add_argument("--robot-glb-up-axis", choices=["y", "z"], default="z")
    parser.add_argument("--robot-target-height", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON.")
    parser.add_argument(
        "--allow-blocked-summary",
        action="store_true",
        help="Allow planning even if --summary-json has status=blocked. Execution is still refused.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = None
    if args.summary_json is not None:
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        if (
            isinstance(summary, dict)
            and summary.get("status") == "blocked"
            and not args.allow_blocked_summary
        ):
            print(
                f"Refusing blocked summary {args.summary_json}; rerun preflight or pass --allow-blocked-summary for planning.",
                file=sys.stderr,
            )
            return 2
    output_root = args.output_root or (args.manifest_json.resolve().parent / "render_jobs")
    manifest = load_render_manifest(args.manifest_json)
    plan_payload = build_render_plans(
        manifest,
        manifest_path=args.manifest_json,
        output_root=output_root,
        scenes_dir=args.scenes_dir,
        tasks_dir=args.tasks_dir,
        render_script=args.render_script,
        python_bin=args.python_bin,
        families=args.family,
        job_ids=args.job_id,
        robot_ids=args.robot_id,
        sensor_names=args.sensor_names,
        limit=args.limit,
        write_inputs=bool(args.write_inputs),
        video_backend=str(args.video_backend),
        device=str(args.device),
        save_depth_maps=bool(args.save_depth_maps),
        save_rgb_frames=bool(args.save_rgb_frames),
        minimal_frames=args.minimal_frames,
        actor_gpu_resident=bool(args.actor_gpu_resident),
        save_actor_metadata=bool(args.save_actor_metadata),
        robot_overlay_script=args.robot_overlay_script,
        robot_glb=args.robot_glb,
        robot_urdf=args.robot_urdf,
        kimodo_smplx_dir=args.kimodo_smplx_dir,
        robot_compose_mode=str(args.robot_compose_mode),
        robot_glb_up_axis=str(args.robot_glb_up_axis),
        robot_target_height=args.robot_target_height,
    )
    if summary is not None:
        plan_payload["summary_status"] = summary.get("status") if isinstance(summary, dict) else None
    if args.json:
        print(json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_plan_text(plan_payload))
    if args.execute:
        if not args.write_inputs:
            print("--execute requires --write-inputs so renderer label JSONs exist.", file=sys.stderr)
            return 2
        return execute_render_plans(plan_payload)
    return 0 if plan_payload["job_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
