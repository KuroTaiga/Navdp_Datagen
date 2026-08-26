#!/usr/bin/env python3
"""Copy the 20260811 Kimodo/STMC human avatar assets to pathGen_lxh.

The rendered output trees are large, so this copies top-level action folders
independently and skips any destination folder whose file count and total bytes
already match the source.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SRC_HOST = "4090_Sun"
DST_HOST = "pathGen_lxh"
DST_BASE = "/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions"

PROMPTS = [
    (
        "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/prompts/expanded_actions.txt",
        f"{DST_BASE}/kimodo/prompts",
    ),
    (
        "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/prompts/stmc_more_actions.txt",
        f"{DST_BASE}/stmc/prompts",
    ),
]

CONTENT_TREES = [
    (
        "kimodo/motionjson",
        "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expandedforkimodo/kimodo",
        f"{DST_BASE}/kimodo/motionjson",
    ),
    (
        "stmc/motionjson",
        "/home/lenovo/TeleNav_RenderPipe/motion_batches/20260811_stmc_kimodo_new_actions/generated_expanded/stmc",
        f"{DST_BASE}/stmc/motionjson",
    ),
]

OUTPUT_TREES = [
    ("kimodo/outputs", "/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_k", f"{DST_BASE}/kimodo/outputs"),
    ("stmc/outputs", "/home/lenovo/TeleNav_RenderPipe/animations_lhmpp_s", f"{DST_BASE}/stmc/outputs"),
]


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
        f"wc -c < {quote(path)} | awk '{{printf \"1 %d\\n\", $1}}'; "
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


def compressed_tar_contents_command(src_dir: str) -> str:
    return (
        "set -euo pipefail; "
        f"cd {quote(src_dir)}; "
        "if command -v pigz >/dev/null 2>&1; then "
        "tar -cf - . | pigz -1 -p 2; "
        "else "
        "tar -cf - . | gzip -1; "
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
    log(f"COPY file {src} -> {dst_dir}/ src_sig={src_sig} dst_sig={dst_sig}")
    remote_run(DST_HOST, f"mkdir -p {quote(dst_dir)}")
    pipe_copy(compressed_tar_command(str(Path(src).parent), Path(src).name), extract_command(dst_dir))
    final_sig = file_signature(DST_HOST, dst)
    if final_sig != src_sig:
        raise RuntimeError(f"signature mismatch for {dst}: source={src_sig} dest={final_sig}")
    return f"DONE file {dst} sig={final_sig}"


def copy_contents(label: str, src_dir: str, dst_dir: str, force: bool = False) -> str:
    src_sig = tree_signature(SRC_HOST, src_dir)
    dst_sig = tree_signature(DST_HOST, dst_dir)
    if not force and src_sig == dst_sig:
        return f"SKIP tree {label} sig={dst_sig}"
    log(f"COPY tree {label}: {src_dir}/ -> {dst_dir}/ src_sig={src_sig} dst_sig={dst_sig}")
    pipe_copy(compressed_tar_contents_command(src_dir), extract_command(dst_dir))
    final_sig = tree_signature(DST_HOST, dst_dir)
    if final_sig != src_sig:
        raise RuntimeError(f"signature mismatch for {label}: source={src_sig} dest={final_sig}")
    return f"DONE tree {label} sig={final_sig}"


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


def verify_summary() -> None:
    cmd = f"""
set -e
base={quote(DST_BASE)}
for d in kimodo/prompts kimodo/motionjson kimodo/outputs stmc/prompts stmc/motionjson stmc/outputs; do
  p="$base/$d"
  printf "%s files=" "$d"
  find "$p" -type f | wc -l
  du -sh "$p"
done
"""
    subprocess.run(["ssh", DST_HOST, cmd], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log(f"START source={SRC_HOST} dest={DST_HOST}:{DST_BASE} jobs={args.jobs}")
    for src, dst in PROMPTS:
        log(copy_file(src, dst, force=args.force))
    for label, src, dst in CONTENT_TREES:
        log(copy_contents(label, src, dst, force=args.force))

    work: list[tuple[str, str, str, str, bool]] = []
    for label, src_parent, dst_parent in OUTPUT_TREES:
        names = remote_lines(SRC_HOST, f"find {quote(src_parent)} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort")
        log(f"QUEUE {label}: {len(names)} folders")
        work.extend((label, src_parent, name, dst_parent, args.force) for name in names)

    completed = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(copy_named_dir, *item) for item in work]
        for future in as_completed(futures):
            completed += 1
            log(f"{completed}/{len(work)} {future.result()}")
    log(f"DONE output folders={len(work)} elapsed={time.time() - start:.1f}s")
    verify_summary()
    log("DONE all assets copied and verified")


if __name__ == "__main__":
    main()
