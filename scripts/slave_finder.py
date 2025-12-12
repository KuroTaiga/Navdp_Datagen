#!/usr/bin/env python3
"""
Slave Finder
=============
Gradio dashboard that polls remote machines over SSH and summarizes their load.

Hosts are read from a .env-style file with blocks such as:

Host worker-1
  HostName 192.168.151.36
  User dongjk
  Password abc

Configuration:
- SLAVE_FINDER_ENV: override path to the .env file (defaults to repo/.env).
- SLAVE_FINDER_REFRESH: auto-refresh interval in seconds (default: 10).
- SLAVE_FINDER_SSH_TIMEOUT: SSH connection/command timeout in seconds (default: 6).
- SLAVE_FINDER_HOST / SLAVE_FINDER_PORT: Gradio bind host/port (defaults: 0.0.0.0 / 7860).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import gradio as gr

try:
    import paramiko
except ImportError as exc:  # pragma: no cover - defensive import check
    raise SystemExit(
        "paramiko is required for SSH support. Install with `pip install paramiko`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = Path(os.environ.get("SLAVE_FINDER_ENV", REPO_ROOT / ".env"))
SSH_TIMEOUT = float(os.environ.get("SLAVE_FINDER_SSH_TIMEOUT", "6"))
REFRESH_SECONDS = float(os.environ.get("SLAVE_FINDER_REFRESH", "10"))

TABLE_HEADERS = [
    "Host",
    "Address",
    "Monitor",
    "Load (1m / 5m / 15m)",
    "CPU cores",
    "Memory (used / total)",
    "Uptime",
    "Status",
]

REMOTE_STATUS_COMMAND = r"""LANG=C
monitor="none"
if command -v btop >/dev/null 2>&1; then
  monitor="btop"
elif command -v vtop >/dev/null 2>&1; then
  monitor="vtop"
fi

load1=$(cut -d' ' -f1 /proc/loadavg)
load5=$(cut -d' ' -f2 /proc/loadavg)
load15=$(cut -d' ' -f3 /proc/loadavg)

cpu_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 0)

mem_total=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)
mem_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)

uptime_out=$(uptime -p 2>/dev/null || uptime)

printf "%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n" \
  "$monitor" "$load1" "$load5" "$load15" "$cpu_count" "$mem_total" "$mem_avail" "$uptime_out"
"""


@dataclass
class HostTarget:
    name: str
    hostname: str
    username: str
    password: str


def parse_env_file(env_path: Path = DEFAULT_ENV_PATH) -> List[HostTarget]:
    """Parse the custom .env file into host targets."""
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing host config file: {path}")

    hosts: List[HostTarget] = []
    current: dict[str, str] = {}

    def finalize_current() -> None:
        if not current:
            return
        missing = [key for key in ("name", "hostname", "username", "password") if key not in current]
        if missing:
            raise ValueError(
                f"Host '{current.get('name', '<unknown>')}' missing fields: {', '.join(missing)}"
            )
        hosts.append(
            HostTarget(
                name=current["name"],
                hostname=current["hostname"],
                username=current["username"],
                password=current["password"],
            )
        )
        current.clear()

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            finalize_current()
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            continue

        key, val = parts
        key_lower = key.lower()
        if key_lower == "host":
            finalize_current()
            current["name"] = val
        elif key_lower == "hostname":
            current["hostname"] = val
        elif key_lower == "user":
            current["username"] = val
        elif key_lower == "password":
            current["password"] = val

    finalize_current()

    if not hosts:
        raise ValueError(f"No host entries found in {path}")

    return hosts


def ssh_exec(host: HostTarget, command: str = REMOTE_STATUS_COMMAND) -> Tuple[str, str, int]:
    """Run a command on the remote host and return (stdout, stderr, exit_code)."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=host.hostname,
            username=host.username,
            password=host.password,
            timeout=SSH_TIMEOUT,
            banner_timeout=SSH_TIMEOUT,
            auth_timeout=SSH_TIMEOUT,
            allow_agent=False,
            look_for_keys=False,
        )
        stdin, stdout, stderr = client.exec_command(command, timeout=SSH_TIMEOUT)
        output = stdout.read().decode()
        err_output = stderr.read().decode()
        exit_status = stdout.channel.recv_exit_status()
    finally:
        client.close()

    return output, err_output, exit_status


def format_memory(mem_total_kb: float, mem_avail_kb: float) -> str:
    """Return a human-readable memory usage string."""
    if mem_total_kb <= 0:
        return "n/a"

    mem_used_kb = max(mem_total_kb - mem_avail_kb, 0.0)
    mem_total_gb = mem_total_kb / (1024 * 1024)
    mem_used_gb = mem_used_kb / (1024 * 1024)
    mem_used_pct = (mem_used_kb / mem_total_kb) * 100 if mem_total_kb else 0.0

    return f"{mem_used_gb:.2f} / {mem_total_gb:.2f} GB ({mem_used_pct:.1f}%)"


def parse_remote_output(host: HostTarget, output: str) -> dict:
    """Parse the newline-delimited payload returned by REMOTE_STATUS_COMMAND."""
    lines = output.strip("\n").splitlines()
    if len(lines) < 8:
        raise ValueError(f"Incomplete response from {host.name}: expected 8 lines, got {len(lines)}")

    monitor, load1, load5, load15, cpu_count, mem_total, mem_avail, uptime = lines[:8]

    def safe_float(text: str) -> float:
        try:
            return float(text)
        except (TypeError, ValueError):
            return 0.0

    load_vals = [safe_float(val) for val in (load1, load5, load15)]
    load_display = " / ".join(f"{val:.2f}" for val in load_vals)

    mem_total_kb = safe_float(mem_total)
    mem_avail_kb = safe_float(mem_avail)

    return {
        "host": host.name,
        "address": host.hostname,
        "monitor": monitor or "none",
        "load": load_display,
        "cpu_count": cpu_count.strip() or "?",
        "memory": format_memory(mem_total_kb, mem_avail_kb),
        "uptime": uptime.strip(),
    }


def fetch_host_status(host: HostTarget) -> dict:
    """Collect status for a single host and normalize fields."""
    output, err_output, exit_code = ssh_exec(host)
    if exit_code != 0:
        detail = err_output.strip() or f"exit code {exit_code}"
        raise RuntimeError(f"{host.name}: remote command failed ({detail})")

    return parse_remote_output(host, output)


def gather_all_statuses() -> Tuple[list[list[str]], str]:
    """Gradio callback to refresh all hosts and build display rows."""
    try:
        hosts = parse_env_file()
    except Exception as exc:  # broad catch to bubble config issues to UI
        message = f"Config error: {exc}"
        return [[ "-", "-", "-", "-", "-", "-", "-", message ]], message

    rows: list[list[str]] = []
    errors: list[str] = []

    for host in hosts:
        try:
            stats = fetch_host_status(host)
            rows.append(
                [
                    stats["host"],
                    stats["address"],
                    stats["monitor"],
                    stats["load"],
                    str(stats["cpu_count"]),
                    stats["memory"],
                    stats["uptime"],
                    "ok",
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive for runtime errors
            msg = f"{host.name}: {exc}"
            rows.append(
                [
                    host.name,
                    host.hostname,
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    f"error: {exc}",
                ]
            )
            errors.append(msg)

    summary = "All hosts refreshed." if not errors else " | ".join(errors)
    return rows, summary


def build_demo(refresh_seconds: float = REFRESH_SECONDS) -> gr.Blocks:
    """Construct the Gradio app layout."""
    with gr.Blocks(title="Slave Finder") as demo:
        gr.Markdown(
            "# Slave Finder\n"
            "Load monitor powered by SSH (prefers btop/vtop when present). "
            "Credentials are pulled from your `.env` host file."
        )

        refresh_button = gr.Button("Refresh now", variant="primary")
        status_table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["str"] * len(TABLE_HEADERS),
            row_count=(1, "dynamic"),
            interactive=False,
        )
        summary = gr.Markdown()

        refresh_button.click(fn=gather_all_statuses, outputs=[status_table, summary])

        # Initial load + optional auto-refresh.
        demo.load(fn=gather_all_statuses, outputs=[status_table, summary])
        if refresh_seconds > 0:
            demo.load(
                fn=gather_all_statuses,
                outputs=[status_table, summary],
                every=refresh_seconds,
            )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue().launch(
        server_name=os.environ.get("SLAVE_FINDER_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("SLAVE_FINDER_PORT", "7860")),
        show_api=False,
    )


if __name__ == "__main__":
    main()
