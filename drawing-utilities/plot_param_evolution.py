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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    records = load_records(args.search_dir)
    print(f"Loaded {len(records)} configs")
    if not records:
        raise SystemExit("No config.json files found")

    out_dir = args.out_dir or (args.search_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = set()
    flattened = []
    for r in records:
        flat = flatten_scalars(r["config"])
        flattened.append(flat)
        params.update(flat.keys())

    for param in sorted(params):
        ys, xs, labels = [], [], []
        for idx, r in enumerate(records):
            if param not in flattened[idx]:
                continue
            ys.append(flattened[idx][param])
            xs.append(len(xs))
            labels.append(f"s{r['stage']}i{r['iter']}c{r['cand']}")

        if not ys:
            continue

        plt.figure(figsize=(14, 4))
        plt.plot(xs, ys, marker="o", linewidth=1)
        plt.xticks(xs, labels, rotation=90, fontsize=6)
        plt.title(param)
        plt.tight_layout()
        out = out_dir / f"{param.replace('.', '_')}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
