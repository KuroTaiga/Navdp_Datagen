#!/usr/bin/env python3
"""
Diagnose likely bottlenecks for a running process by sampling system stats.

Examples:
  ./diagnose_bottleneck.py --pid 12345
  ./diagnose_bottleneck.py --match parallel_render_paths.py --duration 120
  ./diagnose_bottleneck.py --system --duration 120
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


HZ = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
CPU_COUNT = os.cpu_count() or 1


@dataclass
class StatSeries:
    values: List[float] = field(default_factory=list)

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        self.values.append(float(value))

    def avg(self) -> Optional[float]:
        if not self.values:
            return None
        return sum(self.values) / len(self.values)

    def min(self) -> Optional[float]:
        if not self.values:
            return None
        return min(self.values)

    def max(self) -> Optional[float]:
        if not self.values:
            return None
        return max(self.values)


@dataclass
class GPUSeries:
    util: StatSeries = field(default_factory=StatSeries)
    mem_util: StatSeries = field(default_factory=StatSeries)
    mem_used: StatSeries = field(default_factory=StatSeries)
    pcie_rx: StatSeries = field(default_factory=StatSeries)
    pcie_tx: StatSeries = field(default_factory=StatSeries)
    mem_total: Optional[float] = None


def fmt_bytes(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(num_bytes)
    for unit in units:
        if abs(val) < 1024 or unit == units[-1]:
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{val:.2f} TB"


def fmt_rate(num_bytes_per_sec: Optional[float]) -> str:
    if num_bytes_per_sec is None:
        return "n/a"
    return f"{fmt_bytes(num_bytes_per_sec)}/s"


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def fmt_float(value: Optional[float], precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def read_meminfo() -> Dict[str, int]:
    info: Dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val = parts[1].strip().split()[0]
            try:
                info[key] = int(val)
            except ValueError:
                continue
    return info


def read_proc_stat() -> List[int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("cpu "):
                parts = line.split()[1:]
                return [int(p) for p in parts]
    return []


def read_proc_pid_stat(pid: int) -> Dict[str, int]:
    path = f"/proc/{pid}/stat"
    with open(path, "r", encoding="utf-8") as handle:
        data = handle.read()
    rparen = data.rfind(")")
    if rparen == -1:
        raise ValueError(f"Unable to parse {path}")
    after = data[rparen + 2 :].split()
    if len(after) < 22:
        raise ValueError(f"Unexpected stat format for pid {pid}")
    return {
        "ppid": int(after[1]),
        "minflt": int(after[7]),
        "cminflt": int(after[8]),
        "majflt": int(after[9]),
        "cmajflt": int(after[10]),
        "utime": int(after[11]),
        "stime": int(after[12]),
        "rss_pages": int(after[21]),
    }


def read_proc_status(pid: int) -> Dict[str, str]:
    status: Dict[str, str] = {}
    with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            status[key.strip()] = val.strip()
    return status


def read_proc_io(pid: int) -> Dict[str, int]:
    data: Dict[str, int] = {}
    with open(f"/proc/{pid}/io", "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                data[key.strip()] = int(val.strip())
            except ValueError:
                continue
    return data


def list_block_devices() -> List[str]:
    try:
        devices = []
        for entry in os.listdir("/sys/block"):
            if not entry:
                continue
            if entry.startswith(("loop", "ram", "fd", "sr")):
                continue
            devices.append(entry)
        return devices
    except FileNotFoundError:
        return []


def read_diskstats(devices: Iterable[str]) -> Tuple[int, int]:
    devices_set = set(devices)
    read_sectors = 0
    write_sectors = 0
    with open("/proc/diskstats", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 14:
                continue
            name = parts[2]
            if devices_set and name not in devices_set:
                continue
            try:
                read_sectors += int(parts[5])
                write_sectors += int(parts[9])
            except ValueError:
                continue
    return read_sectors * 512, write_sectors * 512


def read_netdev(include_loopback: bool) -> Tuple[int, int]:
    rx_bytes = 0
    tx_bytes = 0
    with open("/proc/net/dev", "r", encoding="utf-8") as handle:
        lines = handle.readlines()[2:]
    for line in lines:
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        iface = iface.strip()
        if iface == "lo" and not include_loopback:
            continue
        fields = rest.split()
        if len(fields) < 9:
            continue
        try:
            rx_bytes += int(fields[0])
            tx_bytes += int(fields[8])
        except ValueError:
            continue
    return rx_bytes, tx_bytes


def get_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read().replace(b"\x00", b" ").strip()
            return raw.decode(errors="replace")
    except FileNotFoundError:
        return ""


def find_pids_by_match(pattern: str) -> List[Tuple[int, str]]:
    matches: List[Tuple[int, str]] = []
    regex = re.compile(pattern)
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        cmdline = get_cmdline(pid)
        if not cmdline:
            continue
        if regex.search(cmdline):
            matches.append((pid, cmdline))
    return matches


def build_ppid_index() -> Dict[int, List[int]]:
    index: Dict[int, List[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            stat = read_proc_pid_stat(pid)
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        ppid = stat.get("ppid")
        if ppid is None:
            continue
        index.setdefault(ppid, []).append(pid)
    return index


def expand_descendants(root_pids: Iterable[int], ppid_index: Dict[int, List[int]]) -> List[int]:
    seen = set(root_pids)
    queue = list(root_pids)
    while queue:
        parent = queue.pop()
        for child in ppid_index.get(parent, []):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
    return sorted(seen)


def resolve_match_targets(
    pattern: str,
    include_children: bool,
    exclude_pids: Iterable[int],
) -> Tuple[List[int], Dict[int, str]]:
    exclude = set(exclude_pids)
    matches = find_pids_by_match(pattern)
    matched_pids: List[int] = []
    cmdlines: Dict[int, str] = {}
    for pid, cmdline in matches:
        if pid in exclude:
            continue
        matched_pids.append(pid)
        cmdlines[pid] = cmdline
    if not matched_pids:
        return [], cmdlines
    targets: List[int]
    if include_children:
        ppid_index = build_ppid_index()
        targets = expand_descendants(matched_pids, ppid_index)
        targets = [pid for pid in targets if pid not in exclude]
    else:
        targets = matched_pids
    return sorted(set(targets)), cmdlines


def format_pid_list(pids: List[int], max_items: int) -> str:
    if len(pids) <= max_items:
        return ", ".join(str(pid) for pid in pids)
    shown = ", ".join(str(pid) for pid in pids[:max_items])
    return f"{shown} (+{len(pids) - max_items} more)"


def parse_smi_value(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value in {"N/A", "Not Supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def detect_nvidia_smi() -> Optional[str]:
    return shutil.which("nvidia-smi")


def query_nvidia_fields() -> List[str]:
    cmd = ["nvidia-smi", "--help-query-gpu"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    fields: List[str] = []
    for line in result.stdout.splitlines():
        candidate = line.strip().split(" ")[0].strip(",")
        if not candidate:
            continue
        if re.fullmatch(r"[A-Za-z0-9_.]+", candidate):
            fields.append(candidate)
    return fields


def read_gpu_stats(query_fields: List[str]) -> List[Dict[str, Optional[float]]]:
    field_arg = ",".join(query_fields)
    cmd = ["nvidia-smi", f"--query-gpu={field_arg}", "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    gpus: List[Dict[str, Optional[float]]] = []
    for raw in result.stdout.strip().splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != len(query_fields):
            continue
        entry: Dict[str, Optional[float]] = {}
        for field, value in zip(query_fields, parts):
            entry[field] = parse_smi_value(value)
        gpus.append(entry)
    return gpus


def parse_memory_mib(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"([0-9]+)\s*MiB", text)
    if not match:
        return None
    return float(match.group(1))


def query_gpu_process_map() -> Dict[int, List[Tuple[int, Optional[float]]]]:
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {}
    proc_map: Dict[int, List[Tuple[int, Optional[float]]]] = {}
    try:
        root = ET.fromstring(result.stdout)
    except ET.ParseError:
        return {}
    for gpu in root.findall("gpu"):
        minor = gpu.findtext("minor_number")
        if minor is None:
            continue
        try:
            gpu_index = int(minor)
        except ValueError:
            continue
        for proc in gpu.findall("processes/process_info"):
            pid_text = proc.findtext("pid")
            if pid_text is None:
                continue
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            used_mem = parse_memory_mib(proc.findtext("used_memory"))
            proc_map.setdefault(pid, []).append((gpu_index, used_mem))
    return proc_map


def choose_gpu_index(
    gpu_series: Dict[int, GPUSeries],
    pids: List[int],
    pid_gpu_map: Dict[int, List[Tuple[int, Optional[float]]]],
    requested_index: Optional[int],
) -> Optional[int]:
    if requested_index is not None:
        return requested_index if requested_index in gpu_series else None
    if pids:
        gpu_ids: List[int] = []
        for pid in pids:
            for gid, _ in pid_gpu_map.get(pid, []):
                gpu_ids.append(gid)
        unique = sorted(set(gpu_ids))
        if len(unique) == 1 and unique[0] in gpu_series:
            return unique[0]
    if not gpu_series:
        return None
    best = None
    best_util = -1.0
    for index, series in gpu_series.items():
        avg_util = series.util.avg() or 0.0
        if avg_util > best_util:
            best_util = avg_util
            best = index
    return best


def assess_bottleneck(summary: Dict[str, Optional[float]], has_gpu: bool, has_pcie: bool) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    candidates: List[Tuple[str, float]] = []

    cpu_pct = summary.get("proc_cpu_pct_total")
    cpu_scope = "Process"
    if cpu_pct is None:
        cpu_pct = summary.get("cpu_busy_pct")
        cpu_scope = "System"
    gpu_util = summary.get("gpu_util")
    vram_pct = summary.get("vram_pct")
    mem_avail_pct = summary.get("mem_avail_pct")
    swap_used_gb = summary.get("swap_used_gb") or 0.0
    majflt_rate = summary.get("majflt_rate")
    iowait_pct = summary.get("cpu_iowait_pct")
    disk_mb_s = summary.get("disk_mb_s") or 0.0
    net_mb_s = summary.get("net_mb_s") or 0.0
    pcie_mb_s = summary.get("pcie_mb_s") or 0.0

    if cpu_pct is not None and cpu_pct >= 80:
        reasons.append(f"{cpu_scope} CPU average {cpu_pct:.1f}% of total CPU capacity.")
        candidates.append(("CPU", min(cpu_pct / 100.0, 1.0)))

    if has_gpu and gpu_util is not None and gpu_util >= 85:
        reasons.append(f"GPU utilization average {gpu_util:.1f}%.")
        candidates.append(("GPU compute", min(gpu_util / 100.0, 1.0)))

    if has_gpu and vram_pct is not None and vram_pct >= 90:
        reasons.append(f"VRAM usage average {vram_pct:.1f}% (near capacity).")
        candidates.append(("VRAM capacity", min(vram_pct / 100.0, 1.0)))

    if mem_avail_pct is not None and mem_avail_pct <= 10:
        reasons.append(f"System MemAvailable average {mem_avail_pct:.1f}% of total.")
        candidates.append(("RAM", min((10 - mem_avail_pct) / 10.0, 1.0)))
    if swap_used_gb > 0.5:
        reasons.append(f"Swap usage {swap_used_gb:.2f} GB.")
        candidates.append(("RAM", 0.6))
    if majflt_rate is not None and majflt_rate >= 5:
        reasons.append(f"Major page faults {majflt_rate:.1f}/s.")
        candidates.append(("RAM", min(majflt_rate / 10.0, 1.0)))

    if iowait_pct is not None and iowait_pct >= 10 and disk_mb_s >= 50:
        reasons.append(f"CPU iowait average {iowait_pct:.1f}% with disk {disk_mb_s:.1f} MB/s.")
        candidates.append(("Disk I/O", min(iowait_pct / 20.0, 1.0)))

    if net_mb_s >= 50:
        reasons.append(f"Network throughput average {net_mb_s:.1f} MB/s.")
        candidates.append(("Network I/O", min(net_mb_s / 100.0, 1.0)))

    if has_pcie and pcie_mb_s >= 2000 and (gpu_util or 0.0) < 60:
        reasons.append(f"PCIe RX+TX average {pcie_mb_s:.1f} (nvidia-smi units) with lower GPU util.")
        candidates.append(("PCIe transfer (RAM<->VRAM)", min(pcie_mb_s / 4000.0, 1.0)))

    if not candidates:
        return "Inconclusive", ["No strong saturation signals detected for CPU/GPU/memory/I/O."]

    candidates.sort(key=lambda item: item[1], reverse=True)
    primary = candidates[0][0]
    return primary, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose runtime bottlenecks for a process.")
    parser.add_argument("--pid", type=int, help="PID to monitor.")
    parser.add_argument(
        "--match",
        help="Regex to select a process by command line if --pid is not provided.",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Sample duration in seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument(
        "--max-pid-list",
        type=int,
        default=10,
        help="Maximum number of PIDs to print when matching multiple processes.",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Sample system-wide stats only (ignore process selection).",
    )
    parser.add_argument(
        "--include-children",
        action="store_true",
        help="Include descendant PIDs of matched processes.",
    )
    parser.add_argument(
        "--no-rescan",
        action="store_true",
        help="Disable periodic rescans for new matching PIDs.",
    )
    parser.add_argument(
        "--rescan-interval",
        type=float,
        default=0.0,
        help="Seconds between rescans when using --match (0 = every sample).",
    )
    parser.add_argument("--gpu-index", type=int, help="Force a specific GPU index.")
    parser.add_argument(
        "--include-loopback",
        action="store_true",
        help="Include loopback traffic in network stats.",
    )
    args = parser.parse_args()

    target_pids: List[int] = []
    pid_cmdlines: Dict[int, str] = {}
    exclude_pids = {os.getpid()}
    if args.system:
        if args.pid is not None or args.match is not None:
            print("WARN: --system ignores --pid/--match selections.", file=sys.stderr)
    elif args.pid is not None:
        target_pids = [args.pid]
        pid_cmdlines[args.pid] = get_cmdline(args.pid)
    else:
        if not args.match:
            print("ERROR: Provide --pid or --match to select a process.", file=sys.stderr)
            return 2
        target_pids, pid_cmdlines = resolve_match_targets(
            args.match,
            args.include_children,
            exclude_pids,
        )
        if not target_pids:
            print(f"ERROR: No process matched '{args.match}'.", file=sys.stderr)
            return 2

    prev_proc_stats: Dict[int, Dict[str, int]] = {}
    prev_proc_io: Dict[int, Dict[str, int]] = {}
    active_pids: List[int] = []
    skipped_pids: List[int] = []
    for pid in target_pids:
        if not os.path.exists(f"/proc/{pid}"):
            skipped_pids.append(pid)
            continue
        try:
            prev_proc_stats[pid] = read_proc_pid_stat(pid)
            prev_proc_io[pid] = read_proc_io(pid)
            active_pids.append(pid)
        except (FileNotFoundError, PermissionError):
            skipped_pids.append(pid)

    if not active_pids and not args.system:
        print("ERROR: No active target processes found.", file=sys.stderr)
        return 2

    rescan_enabled = args.match is not None and not args.no_rescan and not args.system
    rescan_interval = args.rescan_interval if args.rescan_interval > 0 else args.interval

    if args.system:
        print("Scope: system-only (no process filtering)")
    elif len(active_pids) == 1:
        pid = active_pids[0]
        cmdline = pid_cmdlines.get(pid) or get_cmdline(pid)
        print(f"Target PID: {pid}")
        print(f"Command: {cmdline or 'unknown'}")
    else:
        rescan_note = ""
        if rescan_enabled:
            rescan_note = f", rescan every {rescan_interval:.1f}s"
        label = "Targets matched"
        if rescan_enabled:
            label = "Targets matched (initial)"
        print(f"{label}: {len(target_pids)} PIDs{rescan_note}")
        print(f"PIDs: {format_pid_list(sorted(active_pids), args.max_pid_list)}")
        if skipped_pids:
            print(
                f"Skipped: {format_pid_list(sorted(skipped_pids), args.max_pid_list)} "
                "(exited or not accessible)"
            )
        print("Process stats are aggregated across matched PIDs.")
    print(f"Sampling: {args.duration:.1f}s @ {args.interval:.1f}s")

    block_devices = list_block_devices()
    prev_cpu = read_proc_stat()
    prev_disk = read_diskstats(block_devices)
    prev_net = read_netdev(args.include_loopback)
    prev_ts = time.monotonic()

    cpu_busy = StatSeries()
    cpu_iowait = StatSeries()
    proc_cpu_pct_total = StatSeries()
    proc_cpu_cores = StatSeries()
    proc_rss_gb = StatSeries()
    proc_majflt_rate = StatSeries()
    proc_active_pids = StatSeries()
    mem_used_pct = StatSeries()
    mem_avail_pct = StatSeries()
    swap_used_gb = StatSeries()
    disk_read_bps = StatSeries()
    disk_write_bps = StatSeries()
    proc_read_bps = StatSeries()
    proc_write_bps = StatSeries()
    net_rx_bps = StatSeries()
    net_tx_bps = StatSeries()

    gpu_series: Dict[int, GPUSeries] = {}
    nvidia_smi = detect_nvidia_smi()
    query_fields: List[str] = []
    has_pcie = False
    if nvidia_smi:
        fields = query_nvidia_fields()
        query_fields = [
            "index",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
        ]
        if "pcie.rx_throughput" in fields and "pcie.tx_throughput" in fields:
            query_fields += ["pcie.rx_throughput", "pcie.tx_throughput"]
            has_pcie = True
        elif "pcie.rx_util" in fields and "pcie.tx_util" in fields:
            query_fields += ["pcie.rx_util", "pcie.tx_util"]
            has_pcie = True

    sample_count = 0
    exited_pids: List[int] = []
    active_pid_set = set(active_pids)
    seen_pids = set(target_pids)
    last_rescan_ts = time.monotonic()
    while True:
        now = time.monotonic()
        elapsed = now - prev_ts
        if elapsed < args.interval:
            time.sleep(args.interval - elapsed)
            continue

        if rescan_enabled and (now - last_rescan_ts) >= rescan_interval:
            refreshed_pids, refreshed_cmdlines = resolve_match_targets(
                args.match,
                args.include_children,
                exclude_pids,
            )
            for pid, cmdline in refreshed_cmdlines.items():
                pid_cmdlines.setdefault(pid, cmdline)
            for pid in refreshed_pids:
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                try:
                    prev_proc_stats[pid] = read_proc_pid_stat(pid)
                    prev_proc_io[pid] = read_proc_io(pid)
                    active_pid_set.add(pid)
                except (FileNotFoundError, PermissionError, ValueError):
                    continue
            last_rescan_ts = now

        curr_cpu = read_proc_stat()
        curr_disk = read_diskstats(block_devices)
        curr_net = read_netdev(args.include_loopback)
        curr_ts = time.monotonic()
        dt = max(curr_ts - prev_ts, 1e-6)

        if prev_cpu and curr_cpu:
            total_delta = sum(curr_cpu) - sum(prev_cpu)
            idle_delta = curr_cpu[3] - prev_cpu[3]
            iowait_delta = (curr_cpu[4] - prev_cpu[4]) if len(curr_cpu) > 4 else 0
            if total_delta > 0:
                cpu_busy.add(100.0 * (total_delta - idle_delta) / total_delta)
                cpu_iowait.add(100.0 * iowait_delta / total_delta)

        if active_pid_set:
            total_cores_used = 0.0
            total_rss_gb = 0.0
            total_majflt_rate = 0.0
            total_read_bps = 0.0
            total_write_bps = 0.0
            active_count = 0

            for pid in list(active_pid_set):
                try:
                    curr_proc = read_proc_pid_stat(pid)
                    curr_io = read_proc_io(pid)
                except FileNotFoundError:
                    active_pid_set.remove(pid)
                    exited_pids.append(pid)
                    continue
                prev_proc = prev_proc_stats.get(pid)
                prev_io = prev_proc_io.get(pid)
                if prev_proc is None or prev_io is None:
                    prev_proc_stats[pid] = curr_proc
                    prev_proc_io[pid] = curr_io
                    continue

                proc_time_delta = (curr_proc["utime"] + curr_proc["stime"]) - (
                    prev_proc["utime"] + prev_proc["stime"]
                )
                proc_seconds = proc_time_delta / HZ
                cores_used = proc_seconds / dt
                total_cores_used += cores_used

                majflt_delta = curr_proc["majflt"] - prev_proc["majflt"]
                total_majflt_rate += majflt_delta / dt

                rss_kb = curr_proc["rss_pages"] * PAGE_SIZE / 1024
                total_rss_gb += rss_kb / (1024 * 1024)

                total_read_bps += (
                    (curr_io.get("read_bytes", 0) - prev_io.get("read_bytes", 0)) / dt
                )
                total_write_bps += (
                    (curr_io.get("write_bytes", 0) - prev_io.get("write_bytes", 0)) / dt
                )

                prev_proc_stats[pid] = curr_proc
                prev_proc_io[pid] = curr_io
                active_count += 1

            if active_count == 0:
                print(
                    "WARN: No active target processes remaining; ending early.",
                    file=sys.stderr,
                )
                break

            proc_cpu_cores.add(total_cores_used)
            proc_cpu_pct_total.add(100.0 * total_cores_used / CPU_COUNT)
            proc_rss_gb.add(total_rss_gb)
            proc_majflt_rate.add(total_majflt_rate)
            proc_read_bps.add(total_read_bps)
            proc_write_bps.add(total_write_bps)
            proc_active_pids.add(active_count)

        meminfo = read_meminfo()
        mem_total_kb = meminfo.get("MemTotal", 0)
        mem_avail_kb = meminfo.get("MemAvailable", 0)
        if mem_total_kb > 0:
            mem_avail_pct.add(100.0 * mem_avail_kb / mem_total_kb)
            mem_used_pct.add(100.0 * (mem_total_kb - mem_avail_kb) / mem_total_kb)
        swap_total_kb = meminfo.get("SwapTotal", 0)
        swap_free_kb = meminfo.get("SwapFree", 0)
        if swap_total_kb > 0:
            swap_used_gb.add((swap_total_kb - swap_free_kb) / (1024 * 1024))

        disk_read_bps.add((curr_disk[0] - prev_disk[0]) / dt)
        disk_write_bps.add((curr_disk[1] - prev_disk[1]) / dt)
        net_rx_bps.add((curr_net[0] - prev_net[0]) / dt)
        net_tx_bps.add((curr_net[1] - prev_net[1]) / dt)

        if query_fields:
            gpu_stats = read_gpu_stats(query_fields)
            for gpu in gpu_stats:
                idx = gpu.get("index")
                if idx is None:
                    continue
                gpu_index = int(idx)
                series = gpu_series.setdefault(gpu_index, GPUSeries())
                series.util.add(gpu.get("utilization.gpu"))
                series.mem_util.add(gpu.get("utilization.memory"))
                series.mem_used.add(gpu.get("memory.used"))
                series.mem_total = gpu.get("memory.total") or series.mem_total
                series.pcie_rx.add(
                    gpu.get("pcie.rx_throughput") or gpu.get("pcie.rx_util")
                )
                series.pcie_tx.add(
                    gpu.get("pcie.tx_throughput") or gpu.get("pcie.tx_util")
                )

        prev_cpu = curr_cpu
        prev_disk = curr_disk
        prev_net = curr_net
        prev_ts = curr_ts
        sample_count += 1

        if (sample_count * args.interval) >= args.duration:
            break

    if sample_count == 0:
        print("ERROR: No samples collected.", file=sys.stderr)
        return 2

    pid_gpu_map = query_gpu_process_map() if nvidia_smi else {}
    seen_pid_list = sorted(seen_pids)
    chosen_gpu = choose_gpu_index(gpu_series, seen_pid_list, pid_gpu_map, args.gpu_index)

    if not args.system:
        print("")
        print("Targets:")
        print(
            f"  seen {len(seen_pids)}, active avg {fmt_float(proc_active_pids.avg(), 1)}, "
            f"max {fmt_float(proc_active_pids.max(), 0)}"
        )
        if exited_pids:
            print(f"  exited during sampling {len(exited_pids)}")

    print("CPU:")
    if not args.system and proc_cpu_cores.avg() is not None:
        print(
            f"  targets avg {fmt_float(proc_cpu_cores.avg())} cores "
            f"({fmt_pct(proc_cpu_pct_total.avg())} of total), "
            f"max {fmt_float(proc_cpu_cores.max())} cores"
        )
    print(f"  system busy avg {fmt_pct(cpu_busy.avg())}, iowait avg {fmt_pct(cpu_iowait.avg())}")

    print("RAM:")
    if not args.system and proc_rss_gb.avg() is not None:
        print(
            f"  targets RSS avg {fmt_float(proc_rss_gb.avg())} GB, "
            f"max {fmt_float(proc_rss_gb.max())} GB"
        )
    print(
        f"  system used avg {fmt_pct(mem_used_pct.avg())}, "
        f"MemAvailable avg {fmt_pct(mem_avail_pct.avg())}, "
        f"swap used avg {fmt_float(swap_used_gb.avg())} GB"
    )

    print("Disk:")
    print(
        f"  system R/W avg {fmt_rate(disk_read_bps.avg())} / {fmt_rate(disk_write_bps.avg())}"
    )
    if not args.system and proc_read_bps.avg() is not None:
        print(
            f"  targets R/W avg {fmt_rate(proc_read_bps.avg())} / "
            f"{fmt_rate(proc_write_bps.avg())}"
        )
        print(
            f"  major faults avg {fmt_float(proc_majflt_rate.avg())}/s, "
            f"iowait avg {fmt_pct(cpu_iowait.avg())}"
        )
    else:
        print(f"  iowait avg {fmt_pct(cpu_iowait.avg())}")

    print("Network:")
    print(f"  system RX/TX avg {fmt_rate(net_rx_bps.avg())} / {fmt_rate(net_tx_bps.avg())}")

    if not gpu_series:
        print("GPU:")
        print("  nvidia-smi not available or no GPU stats collected.")
    else:
        if chosen_gpu is None:
            chosen_gpu = next(iter(gpu_series.keys()))
        gpu = gpu_series[chosen_gpu]
        vram_pct = None
        if gpu.mem_total and gpu.mem_used.avg() is not None:
            vram_pct = 100.0 * (gpu.mem_used.avg() / gpu.mem_total)
        print(f"GPU {chosen_gpu}:")
        print(
            f"  util avg {fmt_pct(gpu.util.avg())}, mem util avg {fmt_pct(gpu.mem_util.avg())}"
        )
        if gpu.mem_total is not None and gpu.mem_used.avg() is not None:
            print(
                f"  VRAM avg {fmt_float(gpu.mem_used.avg(), 1)} / "
                f"{fmt_float(gpu.mem_total, 1)} MiB "
                f"({fmt_pct(vram_pct)})"
            )
        elif gpu.mem_total is not None:
            print(f"  VRAM total {fmt_float(gpu.mem_total, 1)} MiB (usage unavailable)")
        if has_pcie:
            pcie_rx = gpu.pcie_rx.avg()
            pcie_tx = gpu.pcie_tx.avg()
            print(
                f"  PCIe RX/TX avg {fmt_float(pcie_rx, 1)} / {fmt_float(pcie_tx, 1)} "
                "(nvidia-smi units)"
            )
        else:
            print("  PCIe RX/TX avg n/a (nvidia-smi pcie fields unavailable)")
        if not args.system:
            proc_gpu_mem: Dict[int, float] = {}
            proc_gpu_pids = set()
            for pid in seen_pid_list:
                for gid, mem in pid_gpu_map.get(pid, []):
                    proc_gpu_pids.add(pid)
                    if mem is None:
                        continue
                    proc_gpu_mem[gid] = proc_gpu_mem.get(gid, 0.0) + mem
            if proc_gpu_mem:
                mem_desc = ", ".join(
                    f"{gid}: {mem:.0f} MiB" for gid, mem in sorted(proc_gpu_mem.items())
                )
                print(f"  targets GPU memory: {mem_desc}")
            elif proc_gpu_pids:
                print(f"  target GPU processes: {len(proc_gpu_pids)}")

    vram_pct_val = None
    gpu_util_val = None
    pcie_mb_s_val = None
    if gpu_series:
        util_vals = [series.util.avg() for series in gpu_series.values() if series.util.avg() is not None]
        if util_vals:
            gpu_util_val = max(util_vals)
        vram_vals: List[float] = []
        for series in gpu_series.values():
            if series.mem_total and series.mem_used.avg() is not None:
                vram_vals.append(100.0 * (series.mem_used.avg() / series.mem_total))
        if vram_vals:
            vram_pct_val = max(vram_vals)
        if has_pcie:
            pcie_vals = [
                (series.pcie_rx.avg() or 0.0) + (series.pcie_tx.avg() or 0.0)
                for series in gpu_series.values()
            ]
            if pcie_vals:
                pcie_mb_s_val = max(pcie_vals)

    summary = {
        "proc_cpu_pct_total": proc_cpu_pct_total.avg(),
        "cpu_busy_pct": cpu_busy.avg(),
        "cpu_iowait_pct": cpu_iowait.avg(),
        "mem_avail_pct": mem_avail_pct.avg(),
        "swap_used_gb": swap_used_gb.avg(),
        "majflt_rate": proc_majflt_rate.avg(),
        "disk_mb_s": ((disk_read_bps.avg() or 0.0) + (disk_write_bps.avg() or 0.0))
        / (1024 * 1024),
        "net_mb_s": ((net_rx_bps.avg() or 0.0) + (net_tx_bps.avg() or 0.0)) / (1024 * 1024),
        "gpu_util": gpu_util_val,
        "vram_pct": vram_pct_val,
        "pcie_mb_s": pcie_mb_s_val,
    }

    result, reasons = assess_bottleneck(summary, bool(gpu_series), has_pcie)
    print("")
    print("Bottleneck assessment:")
    print(f"  Result: {result}")
    for reason in reasons:
        print(f"  - {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
