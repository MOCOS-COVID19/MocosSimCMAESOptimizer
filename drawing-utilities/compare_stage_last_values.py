#!/usr/bin/env python3
"""Compare the last configs from stage_10m and stage_18m for every scalar parameter."""

import argparse
import csv
import glob
import json
import math
import re
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple


def flatten_scalars(obj, prefix: str = "") -> Dict[str, float]:
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


def load_last_config(stage_dir: Path) -> Tuple[Dict, Dict[str, object]]:
    pattern = str(stage_dir / "**" / "config.json")
    records = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        s_iter = re.search(r"iter_(\d+)", path)
        s_cand = re.search(r"cand_(\d+)", path)
        if not (s_iter and s_cand):
            continue
        records.append((int(s_iter.group(1)), int(s_cand.group(1)), Path(path)))
    if not records:
        raise SystemExit(f"No config.json files found under {stage_dir}")
    records.sort()
    iter_idx, cand_idx, config_path = records[-1]
    cfg = json.loads(config_path.read_text())
    return cfg, {
        "path": config_path,
        "iter": iter_idx,
        "cand": cand_idx,
    }


def format_scalar(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def format_table(headers, rows) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return output.getvalue().strip()


def main():
    parser = argparse.ArgumentParser(
        description="Compare the last scalar values from stage_10m and stage_18m."
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        required=True,
        help="Root directory containing the stage_10m and stage_18m directories.",
    )
    parser.add_argument(
        "--stage-10m-dir",
        type=Path,
        default=None,
        help="Override the default stage_10m path inside the search directory.",
    )
    parser.add_argument(
        "--stage-18m-dir",
        type=Path,
        default=None,
        help="Override the default stage_18m path inside the search directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="If set, write the table output to this file instead of stdout.",
    )

    args = parser.parse_args()
    base_dir = args.search_dir.resolve()
    stage_10m = (args.stage_10m_dir or base_dir / "stage_10m").resolve()
    stage_18m = (args.stage_18m_dir or base_dir / "stage_18m").resolve()

    if not stage_10m.is_dir():
        raise SystemExit(f"{stage_10m} does not exist or is not a directory")
    if not stage_18m.is_dir():
        raise SystemExit(f"{stage_18m} does not exist or is not a directory")

    config10, meta10 = load_last_config(stage_10m)
    config18, meta18 = load_last_config(stage_18m)
    flat10 = flatten_scalars(config10)
    flat18 = flatten_scalars(config18)

    headers = ["param", "stage_10m_last", "stage_18m_last", "diff"]
    rows = []
    params = sorted(set(flat10.keys()) | set(flat18.keys()))
    for param in params:
        val10 = flat10.get(param)
        val18 = flat18.get(param)
        diff = None
        if val10 is not None and val18 is not None:
            diff = abs(val10 - val18)
        rows.append([param, format_scalar(val10), format_scalar(val18), format_scalar(diff)])

    report = [
        f"stage_10m last config: {meta10['path']} (iter={meta10['iter']}, cand={meta10['cand']})",
        f"stage_18m last config: {meta18['path']} (iter={meta18['iter']}, cand={meta18['cand']})",
        "",
        format_table(headers, rows),
    ]
    output = "\n".join(report)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
