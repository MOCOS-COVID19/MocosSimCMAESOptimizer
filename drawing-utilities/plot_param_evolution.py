#!/usr/bin/env python3
"""Plot evolution of all scalar parameters across config.json files in a search tree."""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DISPLAY_PARAM_ALIASES = {
    "imported_cases[0].params.num_infections": "imported_cases[0]_params_num_infections",
    "imported_cases[1].params.time_limit": "imported_cases[1]_params_time_limit",
    "imported_cases[1].params.frequency": "imported_cases[1]_params_frequency",
    "imported_cases[2].params.time_limit": "imported_cases[2]_params_time_limit",
    "imported_cases[2].params.frequency": "imported_cases[2]_params_frequency",
    "imported_cases[3].params.time_limit": "imported_cases[3]_params_time_limit",
    "imported_cases[3].params.frequency": "imported_cases[3]_params_frequency",
}

PARAM_ALTERNATIVES = {
    "imported_cases[0].params.num_infections": ["imported_cases[1].params.num_infections"],
    "imported_cases[1].params.time_limit": ["imported_cases[2].params.time_limit"],
    "imported_cases[1].params.frequency": ["imported_cases[2].params.frequency"],
    "imported_cases[2].params.time_limit": ["imported_cases[3].params.time_limit"],
    "imported_cases[2].params.frequency": ["imported_cases[3].params.frequency"],
    "imported_cases[3].params.time_limit": ["imported_cases[4].params.time_limit"],
    "imported_cases[3].params.frequency": ["imported_cases[4].params.frequency"],
}


def get_nested(d, path):
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur


def flatten_scalars(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_scalars(v, key))
    elif isinstance(obj, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in obj):
            for idx, x in enumerate(obj):
                out[f"{prefix}[{idx}]"] = float(x)
        elif prefix.endswith("imported_cases") and all(isinstance(x, dict) for x in obj):
            for idx, x in enumerate(obj):
                out.update(flatten_scalars(x, f"{prefix}[{idx}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj):
        out[prefix] = float(obj)
    return out


def load_records(search_dir: Path):
    records = []
    files = sorted(glob.glob(str(search_dir / "**" / "config.json"), recursive=True))
    for f in files:
        try:
            cfg = json.load(open(f))
            m = re.search(r"stage_(\d+)", f)
            i = re.search(r"iter_(\d+)", f)
            c = re.search(r"cand_(\d+)", f)
            if not (m and i and c):
                continue
            records.append(
                {
                    "stage": int(m.group(1)),
                    "iter": int(i.group(1)),
                    "cand": int(c.group(1)),
                    "path": f,
                    "config": cfg,
                }
            )
        except Exception:
            pass
    return sorted(records, key=lambda r: (r["stage"], r["iter"], r["cand"]))


def load_optimized_paths(config_path: Path):
    cfg = json.loads(config_path.read_text())
    paths = set(cfg.get("scalar_bounds", {}).keys())
    paths.update(cfg.get("temporal_bounds", {}).keys())
    return paths


def display_name(param: str) -> str:
    return DISPLAY_PARAM_ALIASES.get(param, param)


def resolve_param(flat, param: str):
    if param in flat:
        return param
    for alt in PARAM_ALTERNATIVES.get(param, []):
        if alt in flat:
            return alt
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    records = load_records(args.search_dir)
    print(f"Loaded {len(records)} configs")
    if not records:
        raise SystemExit("No config.json files found")

    optimized = load_optimized_paths(Path(__file__).resolve().parents[1] / "optimizer_config.json")

    out_dir = args.out_dir or (args.search_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = set(DISPLAY_PARAM_ALIASES.keys())
    flattened = []
    for r in records:
        flat = flatten_scalars(r["config"])
        flattened.append(flat)
        params.update(flat.keys())

    plotted = []
    for param in sorted(params):
        source_keys = [param, *PARAM_ALTERNATIVES.get(param, [])]
        if not any(
            key in optimized or any(key.startswith(x + "[") for x in optimized)
            for key in source_keys
        ):
            continue
        ys, xs = [], []
        tick_pos, tick_lab = [], []
        stage_marks = []
        last_stage = None
        stage_iter_seen = set()
        for idx, r in enumerate(records):
            resolved = resolve_param(flattened[idx], param)
            if resolved is None:
                continue
            if last_stage is None or r["stage"] != last_stage:
                stage_marks.append(len(xs))
                last_stage = r["stage"]
            ys.append(flattened[idx][resolved])
            xs.append(len(xs))
            if (r["stage"], r["iter"]) not in stage_iter_seen:
                stage_iter_seen.add((r["stage"], r["iter"]))
                tick_pos.append(len(xs) - 1)
                if r["iter"] == 1:
                    tick_lab.append(f"stage{r['stage']}/{r['iter']}")
                else:
                    tick_lab.append(f"{r['stage']}/{r['iter']}")

        if not ys:
            continue

        plt.figure(figsize=(14, 4))
        plt.plot(xs, ys, marker="o", linewidth=1)
        for m in stage_marks[1:]:
            plt.axvline(m - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.xticks(tick_pos, tick_lab, rotation=45, ha="right", fontsize=7)
        plt.title(display_name(param))
        plt.tight_layout()
        out = out_dir / f"{display_name(param).replace('.', '_')}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        plotted.append(display_name(param))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
