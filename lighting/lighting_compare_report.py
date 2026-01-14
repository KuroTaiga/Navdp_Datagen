#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_stats(payload: dict, *, variant: str) -> tuple[dict, str]:
    if variant == "base" and isinstance(payload.get("base"), dict):
        return payload["base"], "base"
    if "luma_mean" in payload:
        return payload, "root"
    if isinstance(payload.get("filtered"), dict):
        return payload["filtered"], "filtered"
    if isinstance(payload.get("base"), dict):
        return payload["base"], "base"
    raise ValueError("Unable to extract luma stats from payload.")


def _extract_meta(payload: dict) -> dict:
    meta_keys = (
        "elapsed_sec",
        "filter_sec",
        "fps_total",
        "fps_filter",
        "files_processed",
        "frames_processed",
    )
    return {key: payload[key] for key in meta_keys if key in payload}


def _delta_stats(base: dict, variant: dict) -> dict:
    keys = ("luma_mean", "luma_std", "luma_p05", "luma_p50", "luma_p95", "luma_log2_mean")
    delta = {}
    for key in keys:
        if key in base and key in variant and base[key] is not None and variant[key] is not None:
            delta[key] = variant[key] - base[key]
    if "luma_mean" in base and "luma_mean" in variant and base["luma_mean"]:
        delta["ratio_mean"] = variant["luma_mean"] / base["luma_mean"]
    if "luma_log2_mean" in base and "luma_log2_mean" in variant:
        delta["delta_ev"] = variant["luma_log2_mean"] - base["luma_log2_mean"]
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lighting reports against a base report.")
    parser.add_argument("--base-json", type=Path, required=True, help="Base lighting JSON report.")
    parser.add_argument("--post-json", type=Path, default=None, help="Post-process lighting JSON report.")
    parser.add_argument("--render-json", type=Path, default=None, help="Render-time lighting JSON report.")
    parser.add_argument("--cl-json", type=Path, default=None, help="Camera-light lighting JSON report.")
    parser.add_argument("--metrics-base", type=Path, default=None, help="Optional metrics summary for base.")
    parser.add_argument("--metrics-render", type=Path, default=None, help="Optional metrics summary for render.")
    parser.add_argument("--metrics-cl", type=Path, default=None, help="Optional metrics summary for CL.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    base_payload = _load_json(args.base_json)
    base_stats, base_source = _extract_stats(base_payload, variant="base")

    variants: dict[str, dict] = {}

    def add_variant(name: str, path: Path | None, metrics_path: Path | None = None) -> None:
        if path is None:
            return
        payload = _load_json(path)
        stats, source = _extract_stats(payload, variant=name)
        entry = {
            "stats": stats,
            "source": source,
            "delta": _delta_stats(base_stats, stats),
            "meta": _extract_meta(payload),
        }
        if metrics_path is not None and metrics_path.exists():
            entry["metrics"] = _load_json(metrics_path)
        variants[name] = entry

    add_variant("post", args.post_json)
    add_variant("render", args.render_json, args.metrics_render)
    add_variant("cl", args.cl_json, args.metrics_cl)

    report = {
        "base": base_stats,
        "base_source": base_source,
        "variants": variants,
    }
    if args.metrics_base is not None and args.metrics_base.exists():
        report["base_metrics"] = _load_json(args.metrics_base)

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = ["variant,luma_mean,luma_p50,luma_p95,delta_ev,ratio_mean"]
        for name, entry in variants.items():
            stats = entry.get("stats", {})
            delta = entry.get("delta", {})
            rows.append(
                ",".join(
                    [
                        name,
                        str(stats.get("luma_mean")),
                        str(stats.get("luma_p50")),
                        str(stats.get("luma_p95")),
                        str(delta.get("delta_ev")),
                        str(delta.get("ratio_mean")),
                    ]
                )
            )
        args.output_csv.write_text("\n".join(rows))


if __name__ == "__main__":
    main()
