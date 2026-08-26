#!/usr/bin/env python3
"""Copy curated default Kimodo/STMC actions from 4090_Sun to 5880."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SRC_HOST = "4090_Sun"
DST_HOST = "5880host"
DST_BASE = "/mnt/DATA/dongjk/navdp_data/human_avatars/20260811_stmc_kimodo_new_actions/use_default"

SRC = {
    "kimodo": {
        "prompt": "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/prompts/expanded_actions.txt",
        "motion": "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expandedforkimodo/kimodo",
        "outputs": "/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_k",
    },
    "stmc": {
        "prompt": "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/prompts/stmc_more_actions.txt",
        "motion": "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expanded/stmc",
        "outputs": "/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_s",
    },
}


def quote(value: object) -> str:
    return shlex.quote(str(value))


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


def remote_capture(host: str, command: str) -> str:
    return subprocess.check_output(["ssh", host, command], text=True)


def remote_run(host: str, command: str) -> None:
    subprocess.run(["ssh", host, command], check=True)


def remote_lines(host: str, command: str) -> list[str]:
    return [line for line in remote_capture(host, command).splitlines() if line]


def file_signature(host: str, path: str) -> tuple[int, int]:
    cmd = (
        f"if [ -f {quote(path)} ]; then "
        f"wc -c < {quote(path)} | python3 -c 'import sys; print(1, int(sys.stdin.read().strip() or 0))'; "
        "else echo '0 0'; fi"
    )
    count, size = remote_capture(host, cmd).strip().split()
    return int(count), int(size)


def tree_signature(host: str, path: str) -> tuple[int, int]:
    cmd = (
        f"if [ -d {quote(path)} ]; then "
        f"find {quote(path)} -type f -printf '%s\\n' "
        "| python3 -c 'import sys; vals=[int(line.strip() or 0) for line in sys.stdin]; print(len(vals), sum(vals))' ; "
        "else echo '0 0'; fi"
    )
    count, size = remote_capture(host, cmd).strip().split()
    return int(count), int(size)


def compressed_tar_command(parent: str, name: str) -> str:
    return (
        "set -euo pipefail; "
        f"cd {quote(parent)}; "
        "if command -v pigz >/dev/null 2>&1; then "
        f"tar -cf - {quote(name)} | pigz -1 -p 2; "
        "else "
        f"tar -cf - {quote(name)} | gzip -1; "
        "fi"
    )


def extract_command(dst_dir: str) -> str:
    return f"set -euo pipefail; mkdir -p {quote(dst_dir)}; gzip -dc | tar -C {quote(dst_dir)} -xf -"


def pipe_copy(src_cmd: str, dst_cmd: str) -> None:
    src = subprocess.Popen(["ssh", SRC_HOST, src_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert src.stdout is not None
    dst = subprocess.Popen(
        ["ssh", DST_HOST, dst_cmd],
        stdin=src.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    src.stdout.close()
    dst_stdout, dst_stderr = dst.communicate()
    src_stderr = src.stderr.read().decode("utf-8", errors="replace") if src.stderr else ""
    src_rc = src.wait()
    if src_rc != 0 or dst.returncode != 0:
        raise RuntimeError(
            f"source rc={src_rc} dest rc={dst.returncode}\n"
            f"source stderr:\n{src_stderr}\n"
            f"dest stderr:\n{dst_stderr}\n"
            f"dest stdout:\n{dst_stdout}"
        )


def copy_file(src: str, dst_dir: str, force: bool = False) -> str:
    dst = f"{dst_dir}/{Path(src).name}"
    src_sig = file_signature(SRC_HOST, src)
    dst_sig = file_signature(DST_HOST, dst)
    if not force and src_sig == dst_sig:
        return f"SKIP file {dst} sig={dst_sig}"
    pipe_copy(compressed_tar_command(str(Path(src).parent), Path(src).name), extract_command(dst_dir))
    final_sig = file_signature(DST_HOST, dst)
    if final_sig != src_sig:
        raise RuntimeError(f"signature mismatch for {dst}: source={src_sig} dest={final_sig}")
    return f"DONE file {dst} sig={final_sig}"


def copy_named_dir(label: str, src_parent: str, name: str, dst_parent: str, force: bool = False) -> str:
    src_dir = f"{src_parent}/{name}"
    dst_dir = f"{dst_parent}/{name}"
    src_sig = tree_signature(SRC_HOST, src_dir)
    dst_sig = tree_signature(DST_HOST, dst_dir)
    if not force and src_sig == dst_sig:
        return f"SKIP dir {label}/{name} sig={dst_sig}"
    start = time.time()
    log(f"COPY dir {label}/{name} src_sig={src_sig} dst_sig={dst_sig}")
    pipe_copy(compressed_tar_command(src_parent, name), extract_command(dst_parent))
    final_sig = tree_signature(DST_HOST, dst_dir)
    elapsed = time.time() - start
    if final_sig != src_sig:
        raise RuntimeError(f"signature mismatch for {label}/{name}: source={src_sig} dest={final_sig}")
    return f"DONE dir {label}/{name} sig={final_sig} elapsed={elapsed:.1f}s"


def load_default_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["decision_seed"] == "use_default"]
    rows.sort(key=lambda row: (row["source"], int(row["id"])))
    return rows


def source_names_by_id(source: str, root: str) -> dict[str, list[str]]:
    names = remote_lines(SRC_HOST, f"find {quote(root)} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort")
    by_id: dict[str, list[str]] = {}
    for name in names:
        if source == "outputs":
            parts = name.split("__", 1)
            if len(parts) != 2 or len(parts[1]) < 4:
                continue
            action_id = parts[1][:3]
        else:
            action_id = name[:3]
        by_id.setdefault(action_id, []).append(name)
    return by_id


def output_names_with_preview(root: str) -> dict[str, list[str]]:
    names = remote_lines(
        SRC_HOST,
        f"for d in {quote(root)}/subject_*__*; do "
        "[ -d \"$d\" ] || continue; "
        "[ -s \"$d/preview.mp4\" ] || continue; "
        "basename \"$d\"; "
        "done | sort",
    )
    by_id: dict[str, list[str]] = {}
    for name in names:
        parts = name.split("__", 1)
        if len(parts) == 2 and len(parts[1]) >= 4:
            by_id.setdefault(parts[1][:3], []).append(name)
    return by_id


def wait_for_space(min_available_gb: int, timeout_seconds: int) -> None:
    start = time.time()
    while True:
        out = remote_capture(
            DST_HOST,
            "python3 - <<'PY'\n"
            "import os\n"
            "st=os.statvfs('/mnt/DATA')\n"
            "print((st.f_bavail * st.f_frsize) // (1024**3))\n"
            "PY",
        ).strip()
        available = int(out)
        if available >= min_available_gb:
            log(f"/mnt/DATA available={available}G; starting copy")
            return
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"/mnt/DATA has only {available}G available after waiting {timeout_seconds}s")
        log(f"waiting for /mnt/DATA space: available={available}G need={min_available_gb}G")
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-csv", type=Path, default=Path("out/human_avatar_20260811/intern_action_decision_seed.csv"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-available-gb", type=int, default=220)
    parser.add_argument("--wait-timeout-seconds", type=int, default=24 * 60 * 60)
    args = parser.parse_args()

    wait_for_space(args.min_available_gb, args.wait_timeout_seconds)
    rows = load_default_rows(args.decision_csv)
    log(f"START defaults={len(rows)} dest={DST_HOST}:{DST_BASE} jobs={args.jobs}")

    remote_run(DST_HOST, f"mkdir -p {quote(DST_BASE)}/manifests")
    subprocess.run(["scp", str(args.decision_csv), f"{DST_HOST}:{DST_BASE}/manifests/intern_action_decision_seed.csv"], check=True)

    work: list[tuple[str, str, str, str, bool]] = []
    for source in ("kimodo", "stmc"):
        log(copy_file(SRC[source]["prompt"], f"{DST_BASE}/{source}/prompts", force=args.force))
        motion_by_id = source_names_by_id("motion", SRC[source]["motion"])
        output_by_id = output_names_with_preview(SRC[source]["outputs"])
        source_rows = [row for row in rows if row["source"] == source]
        log(f"QUEUE {source}: {len(source_rows)} default actions")
        for row in source_rows:
            action_id = row["id"]
            for name in motion_by_id.get(action_id, []):
                work.append((f"{source}/motionjson", SRC[source]["motion"], name, f"{DST_BASE}/{source}/motionjson", args.force))
            for name in output_by_id.get(action_id, []):
                work.append((f"{source}/outputs", SRC[source]["outputs"], name, f"{DST_BASE}/{source}/outputs", args.force))

    completed = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(copy_named_dir, *item) for item in work]
        for future in as_completed(futures):
            completed += 1
            log(f"{completed}/{len(work)} {future.result()}")
    log(f"DONE default action copy tasks={len(work)} elapsed={time.time() - start:.1f}s")
    remote_run(DST_HOST, f"find {quote(DST_BASE)} -maxdepth 3 -type d | sort | sed -n '1,120p'")


if __name__ == "__main__":
    main()
