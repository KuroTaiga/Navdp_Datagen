#!/usr/bin/env python3
"""Create curated action group folders on pathGen_lxh.

The raw copied assets stay under:
  /team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/{kimodo,stmc}

This script creates browseable grouped folders with symlinks:
  grouped_actions/{use_default,contextual,reject_default}/{kimodo,stmc}
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from collections import Counter
from pathlib import Path


SRC_HOST = "4090_Sun"
DST_HOST = "pathGen_lxh"
DST_BASE = "/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions"
GROUP_ROOT = f"{DST_BASE}/grouped_actions"

SOURCE_ROOTS = {
    "kimodo": {
        "motion": "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expandedforkimodo/kimodo",
        "outputs": "/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_k",
        "dst_motion": f"{DST_BASE}/kimodo/motionjson",
        "dst_outputs": f"{DST_BASE}/kimodo/outputs",
    },
    "stmc": {
        "motion": "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expanded/stmc",
        "outputs": "/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_s",
        "dst_motion": f"{DST_BASE}/stmc/motionjson",
        "dst_outputs": f"{DST_BASE}/stmc/outputs",
    },
}

DECISIONS = ("use_default", "contextual", "reject_default")


def quote(value: object) -> str:
    return shlex.quote(str(value))


def remote_capture(host: str, command: str) -> str:
    return subprocess.check_output(["ssh", host, command], text=True)


def run_ssh_script(host: str, script: str) -> None:
    subprocess.run(["ssh", host, "bash -s"], input=script, text=True, check=True)


def remote_lines(host: str, command: str) -> list[str]:
    return [line for line in remote_capture(host, command).splitlines() if line]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (row["decision_seed"], row["source"], int(row["id"])))
    return rows


def names_by_action_id(names: list[str], source_type: str) -> dict[str, list[str]]:
    by_id: dict[str, list[str]] = {}
    for name in names:
        if source_type == "outputs":
            parts = name.split("__", 1)
            if len(parts) != 2 or len(parts[1]) < 4:
                continue
            action_id = parts[1][:3]
        else:
            action_id = name[:3]
            if not action_id.isdigit():
                continue
        by_id.setdefault(action_id, []).append(name)
    return by_id


def source_motion_names(root: str) -> dict[str, list[str]]:
    names = remote_lines(SRC_HOST, f"find {quote(root)} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort")
    return names_by_action_id(names, "motion")


def source_output_names(root: str) -> dict[str, list[str]]:
    names = remote_lines(
        SRC_HOST,
        f"for d in {quote(root)}/subject_*__*; do "
        "[ -d \"$d\" ] || continue; "
        "[ -s \"$d/preview.mp4\" ] || continue; "
        "basename \"$d\"; "
        "done | sort",
    )
    return names_by_action_id(names, "outputs")


def write_group_csvs(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    fieldnames = list(rows[0].keys()) if rows else []
    for decision in DECISIONS:
        path = out_dir / f"{decision}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row for row in rows if row["decision_seed"] == decision)
        written.append(path)
    return written


def build_link_script(
    rows: list[dict[str, str]],
    motion_by_source: dict[str, dict[str, list[str]]],
    output_by_source: dict[str, dict[str, list[str]]],
) -> tuple[str, list[dict[str, str]]]:
    script_lines = [
        "set -euo pipefail",
        f"mkdir -p {quote(GROUP_ROOT)}/manifests",
    ]
    manifest_rows: list[dict[str, str]] = []

    for decision in DECISIONS:
        for source in ("kimodo", "stmc"):
            base = f"{GROUP_ROOT}/{decision}/{source}"
            script_lines.extend(
                [
                    f"mkdir -p {quote(base)}/motionjson {quote(base)}/outputs {quote(base)}/previews",
                    f"rm -f {quote(base)}/README.txt",
                    f"printf '%s\\n' {quote(f'{decision} {source} action links; see manifests/actions.csv')} > {quote(base)}/README.txt",
                ]
            )

    for row in rows:
        source = row["source"]
        decision = row["decision_seed"]
        action_id = row["id"]
        dst_roots = SOURCE_ROOTS[source]
        base = f"{GROUP_ROOT}/{decision}/{source}"

        for name in motion_by_source[source].get(action_id, []):
            target = f"{dst_roots['dst_motion']}/{name}"
            link = f"{base}/motionjson/{name}"
            script_lines.append(f"ln -sfn {quote(target)} {quote(link)}")
            manifest_rows.append(
                {
                    "source": source,
                    "id": action_id,
                    "code": row["code"],
                    "decision": decision,
                    "asset_type": "motionjson",
                    "link_path": link,
                    "target_path": target,
                    "prompt": row["prompt"],
                }
            )

        for name in output_by_source[source].get(action_id, []):
            target = f"{dst_roots['dst_outputs']}/{name}"
            output_link = f"{base}/outputs/{name}"
            preview_link = f"{base}/previews/{name}.mp4"
            script_lines.append(f"ln -sfn {quote(target)} {quote(output_link)}")
            script_lines.append(f"ln -sfn {quote(target + '/preview.mp4')} {quote(preview_link)}")
            manifest_rows.append(
                {
                    "source": source,
                    "id": action_id,
                    "code": row["code"],
                    "decision": decision,
                    "asset_type": "outputs",
                    "link_path": output_link,
                    "target_path": target,
                    "prompt": row["prompt"],
                }
            )
            manifest_rows.append(
                {
                    "source": source,
                    "id": action_id,
                    "code": row["code"],
                    "decision": decision,
                    "asset_type": "preview",
                    "link_path": preview_link,
                    "target_path": f"{target}/preview.mp4",
                    "prompt": row["prompt"],
                }
            )

    return "\n".join(script_lines) + "\n", manifest_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "id", "code", "decision", "asset_type", "link_path", "target_path", "prompt"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upload_files(files: list[Path], remote_dir: str) -> None:
    subprocess.run(["ssh", DST_HOST, f"mkdir -p {quote(remote_dir)}"], check=True)
    for path in files:
        subprocess.run(["scp", str(path), f"{DST_HOST}:{remote_dir}/{path.name}"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-csv", type=Path, default=Path("out/human_avatar_20260811/intern_action_decision_seed.csv"))
    parser.add_argument("--local-out-dir", type=Path, default=Path("out/human_avatar_20260811/pathgen_grouping"))
    args = parser.parse_args()

    rows = load_rows(args.decision_csv)
    if not rows:
        raise SystemExit(f"no rows found in {args.decision_csv}")

    motion_by_source = {source: source_motion_names(roots["motion"]) for source, roots in SOURCE_ROOTS.items()}
    output_by_source = {source: source_output_names(roots["outputs"]) for source, roots in SOURCE_ROOTS.items()}

    link_script, manifest_rows = build_link_script(rows, motion_by_source, output_by_source)
    run_ssh_script(DST_HOST, link_script)

    manifest_path = args.local_out_dir / "pathgen_group_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    group_csvs = write_group_csvs(rows, args.local_out_dir)
    upload_files([args.decision_csv, manifest_path, *group_csvs], f"{GROUP_ROOT}/manifests")

    counts = Counter((row["decision_seed"], row["source"]) for row in rows)
    print(f"group_root={DST_HOST}:{GROUP_ROOT}")
    for decision in DECISIONS:
        print(
            f"{decision}: kimodo={counts[(decision, 'kimodo')]} "
            f"stmc={counts[(decision, 'stmc')]}"
        )
    print(f"manifest_rows={len(manifest_rows)} local_manifest={manifest_path}")


if __name__ == "__main__":
    main()
