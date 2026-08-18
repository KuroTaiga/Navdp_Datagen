#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Mapping


DEFAULT_HOST = "pathGen_lxh"
DEFAULT_REMOTE_ROOT = "/private_lxh/dongjk/navdata/mass_generation_runs/formal_fast_v1"
DEFAULT_OUTPUT_ROOT = Path("out/formal_fast_remote_smoke")
DEFAULT_SEED = 20260817
DEFAULT_EXPECTED_MISSIONS = 500


REMOTE_PROBE = r"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

cfg = json.loads(sys.stdin.read())
root = Path(cfg["remote_root"])
rng = random.Random(int(cfg["seed"]))
expected = int(cfg["expected_missions"])
examples_per_combo = max(0, int(cfg["examples_per_combo"]))
sample_mode = str(cfg["sample_mode"])


def mission_jsons(json_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in json_dir.glob("*.json")
        if not path.name.endswith("_cornercase_metadata.json")
    )


def best_json_dir(scene_dir: Path) -> tuple[Path | None, list[Path], list[Path]]:
    best_dir = None
    best_missions: list[Path] = []
    best_metadata: list[Path] = []
    for candidate in sorted(scene_dir.rglob("jsons")):
        missions = mission_jsons(candidate)
        metadata = sorted(candidate.glob("*_cornercase_metadata.json"))
        if len(missions) > len(best_missions):
            best_dir = candidate
            best_missions = missions
            best_metadata = metadata
    return best_dir, best_missions, best_metadata


def inspect_scene(scene_dir: Path) -> dict:
    json_dir, missions, metadata = best_json_dir(scene_dir)
    scene_metadata_files = [
        str(path)
        for path in (
            scene_dir / "mass_example_manifest.json",
            scene_dir / "mass_generation_report.json",
            scene_dir / "mass_generation_report.md",
            scene_dir / "mass_generation_progress.json",
        )
        if path.is_file()
    ]
    examples = []
    for mission in missions[:examples_per_combo]:
        metadata_path = json_dir / f"{mission.stem}_cornercase_metadata.json" if json_dir else None
        examples.append(
            {
                "mission_json": str(mission),
                "metadata_json": str(metadata_path) if metadata_path and metadata_path.is_file() else None,
            }
        )
    return {
        "scene": scene_dir.name,
        "scene_dir": str(scene_dir),
        "json_dir": str(json_dir) if json_dir else None,
        "json_rel_dir": str(json_dir.relative_to(scene_dir)) if json_dir else None,
        "scene_metadata_files": scene_metadata_files,
        "mission_count": len(missions),
        "metadata_count": len(metadata),
        "first_mission_json": str(missions[0]) if missions else None,
        "first_metadata_json": str(metadata[0]) if metadata else None,
        "examples": examples,
    }


started = time.perf_counter()
families = sorted(path for path in root.iterdir() if path.is_dir())
samples = []
failures = []
source_summaries = []

for family_dir in families:
    sources = sorted(path for path in family_dir.iterdir() if path.is_dir())
    for source_dir in sources:
        scene_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
        order = list(scene_dirs)
        rng.shuffle(order)

        selected = None
        checked = 0
        partial_preview = None
        if sample_mode == "random":
            if order:
                checked = 1
                selected = inspect_scene(order[0])
        else:
            for scene_dir in order:
                checked += 1
                inspected = inspect_scene(scene_dir)
                complete = (
                    inspected["mission_count"] == expected
                    and inspected["metadata_count"] == expected
                )
                if complete:
                    selected = inspected
                    break
                if partial_preview is None:
                    partial_preview = inspected

        if selected is None and partial_preview is not None:
            selected = partial_preview

        record = {
            "family": family_dir.name,
            "source": source_dir.name,
            "source_scene_count": len(scene_dirs),
            "candidates_checked": checked,
            "expected_missions": expected,
        }
        if selected is not None:
            record.update(selected)
        else:
            record.update(
                {
                    "scene": None,
                    "scene_dir": None,
                    "json_dir": None,
                    "json_rel_dir": None,
                    "scene_metadata_files": [],
                    "mission_count": 0,
                    "metadata_count": 0,
                    "first_mission_json": None,
                    "first_metadata_json": None,
                    "examples": [],
                }
            )

        record["ok"] = (
            record["mission_count"] == expected
            and record["metadata_count"] == expected
        )
        if not record["ok"]:
            failures.append(record)
        samples.append(record)
        source_summaries.append(
            {
                "family": family_dir.name,
                "source": source_dir.name,
                "source_scene_count": len(scene_dirs),
                "candidates_checked": checked,
            }
        )

elapsed = time.perf_counter() - started
summary = {
    "schema_version": "navdp_formal_fast_remote_smoke.v0.1",
    "remote_root": str(root),
    "seed": int(cfg["seed"]),
    "sample_mode": sample_mode,
    "expected_missions": expected,
    "examples_per_combo": examples_per_combo,
    "family_count": len(families),
    "sample_count": len(samples),
    "failure_count": len(failures),
    "ok": not failures,
    "remote_elapsed_sec": elapsed,
    "samples": samples,
    "failures": failures,
    "source_summaries": source_summaries,
}
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample pathGen_lxh formal_fast_v1 MassGen outputs for a structural "
            "and efficiency smoke test."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-missions", type=int, default=DEFAULT_EXPECTED_MISSIONS)
    parser.add_argument(
        "--sample-mode",
        choices=["complete", "random"],
        default="complete",
        help=(
            "complete shuffles scenes and picks the first one with the expected "
            "mission and metadata count; random checks the first random scene."
        ),
    )
    parser.add_argument(
        "--examples-per-combo",
        type=int,
        default=1,
        help="Number of mission JSON examples to download per family/source sample.",
    )
    parser.add_argument(
        "--download-mode",
        choices=["scene-jsons", "examples", "none"],
        default="scene-jsons",
        help=(
            "scene-jsons downloads the selected scene's full jsons tree plus mass_* "
            "reports; examples downloads only examples-per-combo JSON pairs; none "
            "runs a count-only probe."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Deprecated alias for --download-mode none.",
    )
    parser.add_argument(
        "--max-remote-elapsed-sec",
        type=float,
        default=0.0,
        help="If >0, fail when the remote probe exceeds this many seconds.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even when sampled combinations do not meet the expected count.",
    )
    return parser.parse_args()


def _safe_component(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("__")
    return "".join(safe).strip("_") or "unnamed"


def _run_remote_probe(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "remote_root": args.remote_root,
        "seed": int(args.seed),
        "expected_missions": int(args.expected_missions),
        "examples_per_combo": int(args.examples_per_combo),
        "sample_mode": args.sample_mode,
    }
    remote_cmd = "python3 -c " + shlex.quote(REMOTE_PROBE)
    completed = subprocess.run(
        ["ssh", args.host, remote_cmd],
        input=json.dumps(config),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "remote probe failed with rc="
            f"{completed.returncode}\nSTDERR:\n{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"remote probe returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("remote probe JSON must be an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _download_destination(
    output_root: Path,
    *,
    family: str,
    source: str,
    scene: str,
    remote_path: str,
) -> Path:
    return (
        output_root
        / "examples"
        / _safe_component(family)
        / _safe_component(source)
        / _safe_component(scene)
        / Path(remote_path).name
    )


def _sample_output_dir(
    output_root: Path,
    *,
    family: str,
    source: str,
    scene: str,
) -> Path:
    return (
        output_root
        / "selected_scenes"
        / _safe_component(family)
        / _safe_component(source)
        / _safe_component(scene)
    )


def _download_examples(
    *,
    host: str,
    output_root: Path,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    downloads: list[dict[str, Any]] = []
    samples = summary.get("samples", [])
    if not isinstance(samples, list):
        return downloads

    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        family = str(sample.get("family") or "unknown_family")
        source = str(sample.get("source") or "unknown_source")
        scene = str(sample.get("scene") or "unknown_scene")
        examples = sample.get("examples", [])
        if not isinstance(examples, list):
            continue
        for idx, example in enumerate(examples):
            if not isinstance(example, Mapping):
                continue
            for role in ("mission_json", "metadata_json"):
                remote_path = example.get(role)
                if not remote_path:
                    continue
                remote_path = str(remote_path)
                dest = _download_destination(
                    output_root,
                    family=family,
                    source=source,
                    scene=scene,
                    remote_path=remote_path,
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                completed = subprocess.run(
                    ["scp", f"{host}:{remote_path}", str(dest)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                elapsed = time.perf_counter() - started
                downloads.append(
                    {
                        "family": family,
                        "source": source,
                        "scene": scene,
                        "example_index": idx,
                        "role": role,
                        "remote_path": remote_path,
                        "local_path": str(dest),
                        "returncode": completed.returncode,
                        "elapsed_sec": elapsed,
                        "ok": completed.returncode == 0 and dest.is_file(),
                        "stderr": completed.stderr.strip(),
                    }
                )
    return downloads


def _tar_members(sample: Mapping[str, Any]) -> list[str]:
    members: list[str] = []
    json_rel_dir = sample.get("json_rel_dir")
    if json_rel_dir:
        members.append(str(json_rel_dir))
    scene_dir = sample.get("scene_dir")
    for remote_path in sample.get("scene_metadata_files", []) or []:
        if not scene_dir:
            continue
        try:
            rel = Path(str(remote_path)).relative_to(str(scene_dir))
        except ValueError:
            continue
        members.append(str(rel))
    return members


def _extract_tarball(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)


def _download_scene_packages(
    *,
    host: str,
    output_root: Path,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    downloads: list[dict[str, Any]] = []
    samples = summary.get("samples", [])
    if not isinstance(samples, list):
        return downloads

    archive_root = output_root / "scene_archives"
    archive_root.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        family = str(sample.get("family") or "unknown_family")
        source = str(sample.get("source") or "unknown_source")
        scene = str(sample.get("scene") or "unknown_scene")
        scene_dir = sample.get("scene_dir")
        if not scene_dir:
            downloads.append(
                {
                    "family": family,
                    "source": source,
                    "scene": scene,
                    "role": "scene_jsons",
                    "remote_path": None,
                    "local_path": None,
                    "returncode": 1,
                    "elapsed_sec": 0.0,
                    "ok": False,
                    "stderr": "missing scene_dir in remote summary",
                }
            )
            continue

        members = _tar_members(sample)
        if not members:
            downloads.append(
                {
                    "family": family,
                    "source": source,
                    "scene": scene,
                    "role": "scene_jsons",
                    "remote_path": str(scene_dir),
                    "local_path": None,
                    "returncode": 1,
                    "elapsed_sec": 0.0,
                    "ok": False,
                    "stderr": "missing json_rel_dir and scene metadata members",
                }
            )
            continue

        archive_name = (
            f"{_safe_component(family)}__{_safe_component(source)}__"
            f"{_safe_component(scene)}.tar.gz"
        )
        archive_path = archive_root / archive_name
        dest_dir = _sample_output_dir(
            output_root,
            family=family,
            source=source,
            scene=scene,
        )
        remote_cmd = " ".join(
            [
                "tar",
                "-C",
                shlex.quote(str(scene_dir)),
                "-czf",
                "-",
                *[shlex.quote(member) for member in members],
            ]
        )
        started = time.perf_counter()
        with archive_path.open("wb") as handle:
            completed = subprocess.run(
                ["ssh", host, remote_cmd],
                stdout=handle,
                stderr=subprocess.PIPE,
                text=False,
                check=False,
            )
        elapsed = time.perf_counter() - started

        extract_ok = False
        extract_error = ""
        if completed.returncode == 0 and archive_path.is_file():
            try:
                _extract_tarball(archive_path, dest_dir)
                extract_ok = True
            except Exception as exc:  # pragma: no cover - defensive reporting
                extract_error = str(exc)

        json_rel_dir = sample.get("json_rel_dir")
        local_json_dir = dest_dir / str(json_rel_dir) if json_rel_dir else None
        mission_count = 0
        metadata_count = 0
        if local_json_dir and local_json_dir.is_dir():
            mission_count = len(
                [
                    path
                    for path in local_json_dir.glob("*.json")
                    if not path.name.endswith("_cornercase_metadata.json")
                ]
            )
            metadata_count = len(list(local_json_dir.glob("*_cornercase_metadata.json")))

        downloads.append(
            {
                "family": family,
                "source": source,
                "scene": scene,
                "role": "scene_jsons",
                "remote_path": str(scene_dir),
                "local_path": str(dest_dir),
                "archive_path": str(archive_path),
                "returncode": completed.returncode,
                "elapsed_sec": elapsed,
                "ok": (
                    completed.returncode == 0
                    and extract_ok
                    and mission_count == int(sample.get("mission_count") or 0)
                    and metadata_count == int(sample.get("metadata_count") or 0)
                ),
                "stderr": completed.stderr.decode("utf-8", errors="replace").strip()
                if completed.stderr
                else extract_error,
                "member_count": len(members),
                "mission_count": mission_count,
                "metadata_count": metadata_count,
            }
        )
    return downloads


def _build_manifest(
    *,
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    downloads: list[dict[str, Any]],
    total_elapsed_sec: float,
) -> dict[str, Any]:
    download_failures = [item for item in downloads if not item.get("ok")]
    remote_elapsed = float(summary.get("remote_elapsed_sec") or 0.0)
    elapsed_ok = (
        float(args.max_remote_elapsed_sec) <= 0.0
        or remote_elapsed <= float(args.max_remote_elapsed_sec)
    )
    structure_ok = bool(summary.get("ok"))
    return {
        "schema_version": "navdp_formal_fast_remote_smoke_manifest.v0.1",
        "host": args.host,
        "remote_root": args.remote_root,
        "output_root": str(args.output_root),
        "seed": int(args.seed),
        "sample_mode": args.sample_mode,
        "expected_missions": int(args.expected_missions),
        "examples_per_combo": int(args.examples_per_combo),
        "download_mode": "none" if bool(args.no_download) else args.download_mode,
        "download_enabled": not bool(args.no_download) and args.download_mode != "none",
        "remote_summary_path": str(args.output_root / "remote_summary.json"),
        "download_report_path": str(args.output_root / "download_report.json"),
        "remote_elapsed_sec": remote_elapsed,
        "total_elapsed_sec": total_elapsed_sec,
        "max_remote_elapsed_sec": float(args.max_remote_elapsed_sec),
        "elapsed_ok": elapsed_ok,
        "structure_ok": structure_ok,
        "download_ok": not download_failures,
        "ok": structure_ok and elapsed_ok and not download_failures,
        "family_count": summary.get("family_count"),
        "sample_count": summary.get("sample_count"),
        "failure_count": summary.get("failure_count"),
        "download_count": len(downloads),
        "download_failure_count": len(download_failures),
    }


def main() -> int:
    args = _parse_args()
    if args.no_download:
        args.download_mode = "none"
    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = _run_remote_probe(args)
    _write_json(args.output_root / "remote_summary.json", summary)

    downloads: list[dict[str, Any]] = []
    if args.download_mode == "scene-jsons":
        downloads = _download_scene_packages(
            host=args.host,
            output_root=args.output_root,
            summary=summary,
        )
    elif args.download_mode == "examples" and int(args.examples_per_combo) > 0:
        downloads = _download_examples(
            host=args.host,
            output_root=args.output_root,
            summary=summary,
        )
    _write_json(args.output_root / "download_report.json", {"downloads": downloads})

    total_elapsed = time.perf_counter() - started
    manifest = _build_manifest(
        args=args,
        summary=summary,
        downloads=downloads,
        total_elapsed_sec=total_elapsed,
    )
    _write_json(args.output_root / "smoke_manifest.json", manifest)

    print(
        "summary_status "
        f"ok={manifest['ok']} "
        f"structure_ok={manifest['structure_ok']} "
        f"samples={manifest['sample_count']} "
        f"failures={manifest['failure_count']} "
        f"downloads={manifest['download_count']} "
        f"remote_elapsed_sec={manifest['remote_elapsed_sec']:.3f}"
    )
    print(f"wrote {args.output_root / 'smoke_manifest.json'}")

    if manifest["ok"] or args.allow_incomplete:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
