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
- SLAVE_FINDER_SHARE: set to any truthy value to enable Gradio public share links.
"""

from __future__ import annotations

import os
import re
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
DEFAULT_ENV_PATH = Path(os.environ.get("SLAVE_FINDER_ENV", REPO_ROOT / "scripts" / ".env"))
SSH_TIMEOUT = float(os.environ.get("SLAVE_FINDER_SSH_TIMEOUT", "6"))
REFRESH_SECONDS = float(os.environ.get("SLAVE_FINDER_REFRESH", "10"))

TABLE_HEADERS = [
    "PC",
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
    pc_label: str


def parse_env_file(env_path: Path = DEFAULT_ENV_PATH) -> List[HostTarget]:
    """Parse the custom .env file into host targets."""
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing host config file: {path}")

    hosts: List[HostTarget] = []
    current: dict[str, str] = {}

    def derive_pc_label(name: str) -> str:
        match = re.search(r"(\d+)", name)
        return match.group(1) if match else name

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
                pc_label=derive_pc_label(current["name"]),
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

    def safe_int(text: str) -> int:
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return 0

    load_vals = [safe_float(val) for val in (load1, load5, load15)]
    load_display = " / ".join(f"{val:.2f}" for val in load_vals)

    cpu_count_int = max(safe_int(cpu_count), 1)
    cpu_pct = (load_vals[0] / cpu_count_int) * 100 if cpu_count_int else 0.0

    mem_total_kb = safe_float(mem_total)
    mem_avail_kb = safe_float(mem_avail)
    mem_used_kb = max(mem_total_kb - mem_avail_kb, 0.0)
    mem_used_pct = (mem_used_kb / mem_total_kb) * 100 if mem_total_kb else 0.0
    usage_pct = max(cpu_pct, mem_used_pct)

    return {
        "host": host.name,
        "address": host.hostname,
        "monitor": monitor or "none",
        "load": load_display,
        "cpu_count": cpu_count.strip() or "?",
        "memory": format_memory(mem_total_kb, mem_avail_kb),
        "uptime": uptime.strip(),
        "cpu_pct": cpu_pct,
        "mem_pct": mem_used_pct,
        "usage_pct": usage_pct,
    }


def fetch_host_status(host: HostTarget) -> dict:
    """Collect status for a single host and normalize fields."""
    output, err_output, exit_code = ssh_exec(host)
    if exit_code != 0:
        detail = err_output.strip() or f"exit code {exit_code}"
        raise RuntimeError(f"{host.name}: remote command failed ({detail})")

    result = parse_remote_output(host, output)
    result["pc"] = host.pc_label
    return result


def gather_all_statuses() -> Tuple[list[list[str]], str, str]:
    """Gradio callback to refresh all hosts and build display rows/visuals."""
    try:
        hosts = parse_env_file()
    except Exception as exc:  # broad catch to bubble config issues to UI
        message = f"Config error: {exc}"
        empty_table = [[ "-", "-", "-", "-", "-", "-", "-", "-", message ]]
        return empty_table, "", message

    rows: list[list[str]] = []
    visuals: list[dict] = []
    errors: list[str] = []

    for host in hosts:
        try:
            stats = fetch_host_status(host)
            rows.append(
                [
                    stats["pc"],
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
            visuals.append(
                {
                    "label": f"{stats['pc']} ({stats['host']})",
                    "pct": stats["usage_pct"],
                }
            )
        except Exception as exc:  # pragma: no cover - defensive for runtime errors
            msg = f"{host.name}: {exc}"
            rows.append(
                [
                    host.pc_label,
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
            visuals.append({"label": f"{host.pc_label} ({host.name})", "pct": 0})

    summary = "All hosts refreshed." if not errors else " | ".join(errors)
    return rows, build_usage_html(visuals), summary


def color_for_pct(pct: float) -> str:
    """Map 0-100% to green->red hex."""
    clamped = max(0.0, min(pct, 100.0))
    # Linear interpolate between green (46, 204, 113) and red (220, 53, 69).
    start = (46, 204, 113)
    end = (220, 53, 69)
    frac = clamped / 100.0
    r = int(start[0] + (end[0] - start[0]) * frac)
    g = int(start[1] + (end[1] - start[1]) * frac)
    b = int(start[2] + (end[2] - start[2]) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_usage_html(visuals: List[dict]) -> str:
    """Create a simple HTML bar list colored red->green based on usage%."""
    if not visuals:
        return ""

    rows = []
    for item in visuals:
        pct = item.get("pct", 0.0) or 0.0
        pct_clamped = max(0.0, min(pct, 100.0))
        color = color_for_pct(pct_clamped)
        width = pct_clamped
        label = item.get("label", "")
        rows.append(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
              <div style="width:120px;font-weight:600;">{label}</div>
              <div style="flex:1;background:#e8e8e8;border-radius:6px;overflow:hidden;height:12px;">
                <div style="width:{width}%;background:{color};height:100%;"></div>
              </div>
              <div style="width:50px;text-align:right;font-family:monospace;">{pct:.0f}%</div>
            </div>
            """
        )

    return "<div style='padding:6px 0;'>" + "\n".join(rows) + "</div>"


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
        usage_visual = gr.HTML()
        summary = gr.Markdown()

        refresh_button.click(fn=gather_all_statuses, outputs=[status_table, usage_visual, summary])

        # Initial load + optional auto-refresh.
        demo.load(fn=gather_all_statuses, outputs=[status_table, usage_visual, summary])
        if refresh_seconds > 0:
            # Older Gradio versions do not support `every=` on load, so fall back to Timer when present.
            try:
                timer = gr.Timer(refresh_seconds, active=True)
                timer.tick(fn=gather_all_statuses, outputs=[status_table, usage_visual, summary])
            except Exception:
                pass

    return demo


def main() -> None:
    demo = build_demo()
    share_env = os.environ.get("SLAVE_FINDER_SHARE", "").lower()
    # Default to sharing unless explicitly disabled.
    share = share_env not in {"0", "false", "no", "off"}

    launch_kwargs = {
        "server_name": os.environ.get("SLAVE_FINDER_HOST", "0.0.0.0"),
        "server_port": int(os.environ.get("SLAVE_FINDER_PORT", "7860")),
        "share": share,
    }

    # Keep compatibility with older/newer Gradio versions that may or may not support show_api.
    try:
        import inspect

        if "show_api" in inspect.signature(gr.Blocks.launch).parameters:
            launch_kwargs["show_api"] = False
    except Exception:
        pass

    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
