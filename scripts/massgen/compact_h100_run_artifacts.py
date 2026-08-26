#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIVE_LOG_NAMES = {
    "driver.log",
    "launcher.log",
    "monitor.log",
    "compactor.log",
    "report_from_monitor.log",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep long H100 MassGen runs inspectable by tail-snapshotting live "
            "logs, gzipping stale oversized logs, and sampling GPU JSONL output."
        )
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--max-log-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--tail-bytes", type=int, default=256 * 1024)
    parser.add_argument("--stale-sec", type=float, default=600.0)
    parser.add_argument("--gpu-sample-stride", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _tail_snapshot(path: Path, *, tail_bytes: int, dry_run: bool) -> dict[str, Any]:
    snapshot = path.with_name(path.name + ".tail")
    action = {
        "action": "tail_snapshot",
        "path": str(path),
        "snapshot": str(snapshot),
        "tail_bytes": int(tail_bytes),
    }
    if dry_run:
        return action
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - tail_bytes))
        data = handle.read()
    marker = (
        f"# Tail snapshot for {path}\n"
        f"# Generated {datetime.now(timezone.utc).isoformat()}\n"
        f"# Source size bytes: {path.stat().st_size}\n\n"
    ).encode("utf-8")
    snapshot.write_bytes(marker + data)
    return action


def _gzip_file(path: Path, *, dry_run: bool) -> dict[str, Any]:
    gz_path = path.with_name(path.name + ".gz")
    action = {
        "action": "gzip",
        "path": str(path),
        "gz_path": str(gz_path),
        "bytes": path.stat().st_size,
    }
    if dry_run:
        return action
    tmp_path = gz_path.with_name(gz_path.name + ".tmp")
    with path.open("rb") as src, gzip.open(tmp_path, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(gz_path)
    path.unlink()
    return action


def _compact_logs(
    root: Path,
    *,
    max_log_bytes: int,
    tail_bytes: int,
    stale_sec: float,
    dry_run: bool,
) -> list[dict[str, Any]]:
    now = time.time()
    actions: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.log")):
        if path.name.endswith(".tail") or path.name.endswith(".gz"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size <= max_log_bytes:
            continue
        actions.append(_tail_snapshot(path, tail_bytes=tail_bytes, dry_run=dry_run))
        stale = (now - stat.st_mtime) >= stale_sec
        if stale and path.name not in LIVE_LOG_NAMES:
            actions.append(_gzip_file(path, dry_run=dry_run))
    return actions


def _sample_gpu_jsonl(root: Path, *, stride: int, dry_run: bool) -> dict[str, Any] | None:
    gpu_path = root / "run_persistent" / "gpu_samples.jsonl"
    if not gpu_path.is_file():
        return None
    stride = max(1, int(stride))
    out_path = root / "monitor" / "gpu_samples_sampled.jsonl"
    action: dict[str, Any] = {
        "action": "sample_gpu_jsonl",
        "path": str(gpu_path),
        "sampled_path": str(out_path),
        "stride": stride,
        "input_lines": 0,
        "output_lines": 0,
    }
    if dry_run:
        return action
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_line = ""
    with gpu_path.open("r", encoding="utf-8", errors="replace") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for index, line in enumerate(src):
            action["input_lines"] += 1
            last_line = line
            if index % stride == 0:
                dst.write(line)
                action["output_lines"] += 1
        if last_line and (action["input_lines"] - 1) % stride != 0:
            dst.write(last_line)
            action["output_lines"] += 1
    return action


def main() -> int:
    args = _parse_args()
    root = args.results_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    actions = _compact_logs(
        root,
        max_log_bytes=max(1, int(args.max_log_bytes)),
        tail_bytes=max(1, int(args.tail_bytes)),
        stale_sec=max(1.0, float(args.stale_sec)),
        dry_run=bool(args.dry_run),
    )
    gpu_action = _sample_gpu_jsonl(
        root,
        stride=max(1, int(args.gpu_sample_stride)),
        dry_run=bool(args.dry_run),
    )
    if gpu_action is not None:
        actions.append(gpu_action)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(root),
        "dry_run": bool(args.dry_run),
        "actions": actions,
    }
    report_path = root / "monitor" / "artifact_compaction_report.json"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
