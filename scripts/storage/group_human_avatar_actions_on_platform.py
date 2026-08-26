#!/usr/bin/env python3
"""Create a filtered human-avatar action group on a platform-local asset root.

This is for H100/formal-test staging after the full Kimodo/STMC asset tree has
already been copied to the platform. It creates browseable symlink groups under:

  <dst-base>/grouped_actions/<group-name>/{kimodo,stmc}/{motionjson,outputs,previews}
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
from pathlib import Path


DEFAULT_HOST = "envtest"
DEFAULT_DST_BASE = "/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions"
DEFAULT_DECISION = "use_default"
DEFAULT_GROUP_NAME = "use_default_no_waving"
DEFAULT_EXCLUDE_REGEX = r"\b(wave|waves|waving)\b"


def quote(value: object) -> str:
    return shlex.quote(str(value))


def remote_capture(host: str, command: str) -> str:
    return subprocess.check_output(["ssh", host, command], text=True)


def remote_lines(host: str, command: str) -> list[str]:
    return [line for line in remote_capture(host, command).splitlines() if line]


def run_ssh_script(host: str, script: str) -> None:
    subprocess.run(["ssh", host, "bash -s"], input=script, text=True, check=True)


def load_rows(csv_path: Path, decision: str, exclude_regex: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    pattern = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
    selected: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []

    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["decision_seed"] != decision:
                continue
            haystack = " ".join([row.get("prompt", ""), row.get("reason_seed", ""), row.get("code", "")])
            if pattern and pattern.search(haystack):
                excluded.append(row)
            else:
                selected.append(row)

    selected.sort(key=lambda row: (row["source"], int(row["id"])))
    excluded.sort(key=lambda row: (row["source"], int(row["id"])))
    return selected, excluded


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


def motion_names(host: str, root: str) -> dict[str, list[str]]:
    names = remote_lines(host, f"find {quote(root)} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort")
    return names_by_action_id(names, "motion")


def output_names(host: str, root: str) -> dict[str, list[str]]:
    names = remote_lines(
        host,
        f"for d in {quote(root)}/subject_*__*; do "
        "[ -d \"$d\" ] || continue; "
        "basename \"$d\"; "
        "done | sort",
    )
    return names_by_action_id(names, "outputs")


def preview_names(host: str, root: str) -> set[str]:
    return set(
        remote_lines(
            host,
            f"for d in {quote(root)}/subject_*__*; do "
            "[ -d \"$d\" ] || continue; "
            "[ -s \"$d/preview.mp4\" ] || continue; "
            "basename \"$d\"; "
            "done | sort",
        )
    )


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upload_files(host: str, files: list[Path], remote_dir: str) -> None:
    subprocess.run(["ssh", host, f"mkdir -p {quote(remote_dir)}"], check=True)
    for path in files:
        subprocess.run(["scp", str(path), f"{host}:{remote_dir}/{path.name}"], check=True)


def build_link_script(
    rows: list[dict[str, str]],
    motion_by_source: dict[str, dict[str, list[str]]],
    output_by_source: dict[str, dict[str, list[str]]],
    preview_by_source: dict[str, set[str]],
    dst_base: str,
    group_root: str,
    group_name: str,
) -> tuple[str, list[dict[str, str]]]:
    script_lines = [
        "set -euo pipefail",
        f"mkdir -p {quote(group_root)}/manifests",
    ]
    manifest_rows: list[dict[str, str]] = []

    for source in ("kimodo", "stmc"):
        base = f"{group_root}/{group_name}/{source}"
        script_lines.extend(
            [
                f"mkdir -p {quote(base)}/motionjson {quote(base)}/outputs {quote(base)}/previews",
                f"find {quote(base)} -mindepth 1 -maxdepth 2 -type l -delete",
                f"rm -f {quote(base)}/README.txt",
                f"printf '%s\\n' {quote(f'{group_name} {source} action links; see manifests/{group_name}_manifest.csv')} > {quote(base)}/README.txt",
            ]
        )

    for row in rows:
        source = row["source"]
        action_id = row["id"]
        base = f"{group_root}/{group_name}/{source}"
        dst_motion = f"{dst_base}/{source}/motionjson"
        dst_outputs = f"{dst_base}/{source}/outputs"

        for name in motion_by_source[source].get(action_id, []):
            target = f"{dst_motion}/{name}"
            link = f"{base}/motionjson/{name}"
            script_lines.append(f"ln -sfn {quote(target)} {quote(link)}")
            manifest_rows.append(
                {
                    "source": source,
                    "id": action_id,
                    "code": row["code"],
                    "decision": group_name,
                    "asset_type": "motionjson",
                    "link_path": link,
                    "target_path": target,
                    "prompt": row["prompt"],
                }
            )

        for name in output_by_source[source].get(action_id, []):
            target = f"{dst_outputs}/{name}"
            output_link = f"{base}/outputs/{name}"
            preview_link = f"{base}/previews/{name}.mp4"
            script_lines.append(f"ln -sfn {quote(target)} {quote(output_link)}")
            manifest_rows.append(
                {
                    "source": source,
                    "id": action_id,
                    "code": row["code"],
                    "decision": group_name,
                    "asset_type": "outputs",
                    "link_path": output_link,
                    "target_path": target,
                    "prompt": row["prompt"],
                }
            )
            if name in preview_by_source[source]:
                script_lines.append(f"ln -sfn {quote(target + '/preview.mp4')} {quote(preview_link)}")
                manifest_rows.append(
                    {
                        "source": source,
                        "id": action_id,
                        "code": row["code"],
                        "decision": group_name,
                        "asset_type": "preview",
                        "link_path": preview_link,
                        "target_path": f"{target}/preview.mp4",
                        "prompt": row["prompt"],
                    }
                )

    return "\n".join(script_lines) + "\n", manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-csv", type=Path, default=Path("out/human_avatar_20260811/intern_action_decision_seed.csv"))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--dst-base", default=DEFAULT_DST_BASE)
    parser.add_argument("--group-root", default="")
    parser.add_argument("--decision", default=DEFAULT_DECISION)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    parser.add_argument("--local-out-dir", type=Path, default=Path("out/human_avatar_20260811/h100_grouping"))
    args = parser.parse_args()

    group_root = args.group_root or f"{args.dst_base}/grouped_actions"
    rows, excluded = load_rows(args.decision_csv, args.decision, args.exclude_regex)
    if not rows:
        raise SystemExit(f"no rows selected from {args.decision_csv}")

    motion_by_source = {
        source: motion_names(args.host, f"{args.dst_base}/{source}/motionjson")
        for source in ("kimodo", "stmc")
    }
    output_by_source = {
        source: output_names(args.host, f"{args.dst_base}/{source}/outputs")
        for source in ("kimodo", "stmc")
    }
    preview_by_source = {
        source: preview_names(args.host, f"{args.dst_base}/{source}/outputs")
        for source in ("kimodo", "stmc")
    }

    link_script, manifest_rows = build_link_script(
        rows,
        motion_by_source,
        output_by_source,
        preview_by_source,
        args.dst_base,
        group_root,
        args.group_name,
    )
    run_ssh_script(args.host, link_script)

    filtered_csv = args.local_out_dir / f"{args.group_name}.csv"
    excluded_csv = args.local_out_dir / f"{args.group_name}_excluded.csv"
    manifest_csv = args.local_out_dir / f"{args.group_name}_manifest.csv"
    row_fieldnames = list(rows[0].keys())
    write_csv(filtered_csv, rows, row_fieldnames)
    write_csv(excluded_csv, excluded, row_fieldnames)
    write_csv(
        manifest_csv,
        manifest_rows,
        ["source", "id", "code", "decision", "asset_type", "link_path", "target_path", "prompt"],
    )
    upload_files(args.host, [filtered_csv, excluded_csv, manifest_csv], f"{group_root}/manifests")

    print(f"group={args.host}:{group_root}/{args.group_name}")
    print(f"selected_actions={len(rows)} excluded_actions={len(excluded)} manifest_rows={len(manifest_rows)}")
    print("excluded_codes=" + ",".join(row["code"] for row in excluded))
    print(f"local_manifest={manifest_csv}")


if __name__ == "__main__":
    main()
