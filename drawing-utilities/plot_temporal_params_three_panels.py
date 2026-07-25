#!/usr/bin/env python3
"""Plot temporal parameter arrays in three panels for 0..104 days."""

import argparse
import glob
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEMPORAL_PARAMS = [
    ("infection_modulation.params.interval_values", "Infection modulation"),
    ("mild_detection_modulation.params.interval_values", "Mild detection modulation"),
    ("tracing_modulation.params.interval_values", "Tracing modulation"),
]


def get_nested(d, path):
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur


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


def flatten_temporal_array(cfg, path):
    try:
        arr = get_nested(cfg, path)
    except Exception:
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for x in arr[:105]:
        if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
            out.append(float(x))
        else:
            out.append(np.nan)
    if len(out) < 105:
        out.extend([np.nan] * (105 - len(out)))
    return np.asarray(out, dtype=float)


def plot_global(records, out: Path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    days = np.arange(105)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(records)))

    for ax, (param, title) in zip(axes[:3], TEMPORAL_PARAMS):
        for idx, r in enumerate(records):
            vals = flatten_temporal_array(r["config"], param)
            if vals is None:
                continue
            ax.plot(days, vals, color=colors[idx], alpha=0.18, linewidth=1)
        ax.set_title(title)
        ax.set_xlim(0, 104)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("value")

    scores = []
    score_days = []
    for r in records:
        cfg = r["config"]
        score = cfg.get("score")
        if score is None:
            score = cfg.get("metrics", {}).get("score") if isinstance(cfg.get("metrics"), dict) else None
        if score is None:
            continue
        scores.append(float(score))
        score_days.append(len(score_days))
    if scores:
        axes[3].plot(score_days, scores, color="#263238", linewidth=1.8, marker="o", markersize=3)
    axes[3].set_title("Score evolution")
    axes[3].set_ylabel("score")
    axes[3].set_xlim(0, max(score_days) if score_days else 1)
    axes[3].grid(True, alpha=0.25)
    axes[3].set_xlabel("run index")

    axes[-1].set_xlabel("day")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_local(records, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    days = np.arange(105)
    by_run = {}
    for r in records:
        key = (r["stage"], r["iter"])
        by_run.setdefault(key, []).append(r)

    keys = sorted(by_run.keys())
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, max(len(keys), 1)))

    for ax, (param, title) in zip(axes, TEMPORAL_PARAMS):
        for idx, key in enumerate(keys):
            entries = sorted(by_run[key], key=lambda r: r["cand"])
            best = entries[0]
            vals = flatten_temporal_array(best["config"], param)
            if vals is None:
                continue
            label = f"stage {key[0]} iter {key[1]}"
            ax.plot(days, vals, color=colors[idx], alpha=0.45, linewidth=1.5, label=label if idx < 8 else None)
        ax.set_title(title)
        ax.set_xlim(0, 104)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("value")
        if keys:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("day")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="Directory for the output PNGs (defaults to search_dir/plots).")
    ap.add_argument("--local", action="store_true", help="Also write a per-run local panel view like plot_gt_vs_sim.py.")
    args = ap.parse_args()

    records = load_records(args.search_dir)
    if not records:
        raise SystemExit("No config.json files found")

    optimized = load_optimized_paths(Path(__file__).resolve().parents[1] / "optimizer_config.json")
    for param, _ in TEMPORAL_PARAMS:
        if param not in optimized:
            raise SystemExit(f"Temporal param not found in optimizer_config.json: {param}")

    if args.out is not None and args.out_dir is not None:
        raise SystemExit("Use either --out or --out-dir, not both")
    if args.out is not None:
        global_out = args.out
    else:
        out_dir = args.out_dir or (args.search_dir / "plots")
        global_out = out_dir / "temporal_params_three_panels.png"
    global_out.parent.mkdir(parents=True, exist_ok=True)

    plot_global(records, global_out)
    if args.local:
        for r in records:
            run_dir = Path(r["path"]).parent
            local_out = run_dir / "temporal_params_three_panels_local.png"
            single_run_records = [x for x in records if x["stage"] == r["stage"] and x["iter"] == r["iter"] and x["cand"] == r["cand"]]
            plot_local(single_run_records, local_out)


if __name__ == "__main__":
    main()
