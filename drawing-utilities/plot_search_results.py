#!/usr/bin/env python3
"""
Plot checkpoint-delta search results from rmse_progress.json.

Panels produced (saved as individual PNGs and one combined figure):
  1. RMSE timeline      – combined_rmse per search step, coloured by month,
                          best-per-month marker highlighted
  2. Per-metric RMSE    – one line per GT target (detections / deaths / hosp)
  3. Identifiability    – heatmap: temporal-param × bucket, colour = |corr|
                          (read from identifiability_month_XX.json files)
  4. Param evolution    – temporal param bucket values for the accepted config
                          at each month end
  5. Window coverage    – Gantt-like chart: x=days, y=month, shaded windows

Usage
-----
    python plot_search_results.py --output-dir /path/to/search_local
    python plot_search_results.py --output-dir /path/to/search_local --show
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_progress(output_dir: Path) -> List[Dict[str, Any]]:
    """Map optimizer_history.json into a search-like record list."""
    p = output_dir / "optimizer_history.json"
    if not p.exists():
        raise FileNotFoundError(f"optimizer_history.json not found in {output_dir}")
    history = json.loads(p.read_text())
    records: List[Dict[str, Any]] = []
    step = 0
    for entry in history:
        step += 1
        rec = {
            "search_step": step,
            "month": entry.get("fit_months"),
            "combined_rmse": entry.get("metrics", {}).get("score"),
            "start_day": 1,
            "end_day": entry.get("fit_months", 0) * 30,
        }
        # copy metric components if present
        for k in ["full_daily", "recent_daily", "cumulative", "peak_height", "peak_timing", "slope"]:
            if k in entry.get("metrics", {}):
                rec[f"rmse_{k}" if k.startswith("full") or k.startswith("recent") else k] = entry["metrics"][k]
        # copy param snapshots if present (best_vector not stored; none available)
        records.append(rec)
    return records


def load_identifiability(output_dir: Path) -> List[Dict[str, Any]]:
    """No identifiability outputs exist in this optimizer; return empty."""
    return []


def load_accepted_months(output_dir: Path) -> List[Dict[str, Any]]:
    """Approximate accepted configs per month using stage_*_best_candidate.json."""
    files = sorted(output_dir.glob("stage_*_best_candidate.json"))
    accepted = []
    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        # extract month number from filename like stage_02_best_candidate.json
        m = re.search(r"stage_(\d+)", f.name)
        month = int(m.group(1)) if m else len(accepted) + 1
        accepted.append({
            "month": month,
            "config": payload,
            "scores": {},
        })
    return accepted


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "rmse_daily_detections":      "Detections",
    "rmse_daily_deaths":          "Deaths",
    "rmse_daily_hospitalizations":"Hospitalizations",
}

TEMPORAL_SHORT = {
    "infection_modulation":       "Infection mod.",
    "mild_detection_modulation":  "Detection mod.",
    "tracing_modulation":         "Tracing mod.",
}

MONTH_CMAP = plt.cm.tab10


def _finite(v):
    if v is None:
        return float("nan")
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Panel 1+2 – RMSE timeline
# ──────────────────────────────────────────────────────────────────────────────

def plot_rmse_timeline(records: List[Dict], ax_combined, ax_metrics):
    steps   = [r.get("search_step", i+1) for i, r in enumerate(records)]
    months  = [r.get("month", 1)         for r in records]
    combined = [_finite(r.get("combined_rmse")) for r in records]

    unique_months = sorted(set(months))
    colours = {m: MONTH_CMAP(i / max(len(unique_months)-1, 1))
               for i, m in enumerate(unique_months)}

    # ── combined RMSE ──
    for m in unique_months:
        idxs = [i for i, mo in enumerate(months) if mo == m]
        xs = [steps[i] for i in idxs]
        ys = [combined[i] for i in idxs]
        ax_combined.scatter(xs, ys, s=18, color=colours[m], alpha=0.7, zorder=3)
        # connect within-month dots
        ax_combined.plot(xs, ys, "-", color=colours[m], alpha=0.35, linewidth=0.8)

    # best per month marker
    for m in unique_months:
        idxs = [i for i, mo in enumerate(months) if mo == m]
        valid = [(combined[i], steps[i]) for i in idxs if not math.isnan(combined[i])]
        if not valid:
            continue
        best_rmse, best_step = min(valid)
        ax_combined.scatter([best_step], [best_rmse], s=80, marker="*",
                            color=colours[m], edgecolors="k", linewidth=0.5, zorder=5)

    ax_combined.set_ylabel("Combined RMSE")
    ax_combined.set_title("RMSE progression (★ = best per month)")
    ax_combined.set_xlabel("Search step")
    legend_patches = [mpatches.Patch(color=colours[m], label=f"Month {m}")
                      for m in unique_months]
    ax_combined.legend(handles=legend_patches, fontsize=7, ncol=3)
    ax_combined.grid(True, alpha=0.3)

    # ── per-metric RMSE ──
    metric_keys = [k for k in METRIC_LABELS if any(k in r for r in records)]
    line_styles = ["-", "--", ":"]
    for li, key in enumerate(metric_keys):
        ys = [_finite(r.get(key)) for r in records]
        ax_metrics.plot(steps, ys, line_styles[li % 3],
                        label=METRIC_LABELS.get(key, key), linewidth=1.2, alpha=0.8)

    ax_metrics.set_ylabel("RMSE per metric")
    ax_metrics.set_xlabel("Search step")
    ax_metrics.set_title("Per-metric RMSE over time")
    ax_metrics.legend(fontsize=8)
    ax_metrics.grid(True, alpha=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 3 – Identifiability heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_identifiability(id_reports: List[Dict], ax):
    if not id_reports:
        ax.text(0.5, 0.5, "No identifiability data yet",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Parameter identifiability")
        return

    # collect all (path, bucket) pairs seen across all months
    all_paths = set()
    all_buckets: Dict[str, set] = {}
    for rep in id_reports:
        for path, info in rep.get("scores", {}).items():
            short = TEMPORAL_SHORT.get(path.split(".")[0], path.split(".")[0])
            all_paths.add(short)
            buckets = all_buckets.setdefault(short, set())
            for k in info.get("buckets", {}).keys():
                buckets.add(int(k))

    sorted_paths = sorted(all_paths)
    # build a label list: "InfMod b0", "InfMod b1", ...
    row_labels = []
    row_keys   = []   # (original_path_key, bucket_idx)
    for short in sorted_paths:
        orig = next((p for p in id_reports[-1].get("scores", {}).keys()
                     if TEMPORAL_SHORT.get(p.split(".")[0], p.split(".")[0]) == short), None)
        for k in sorted(all_buckets.get(short, set())):
            row_labels.append(f"{short[:12]} b{k}")
            row_keys.append((orig, str(k)))

    if not row_labels:
        ax.text(0.5, 0.5, "No bucket data", ha="center", va="center",
                transform=ax.transAxes)
        return

    months = [r.get("month", i+1) for i, r in enumerate(id_reports)]
    matrix = np.full((len(row_labels), len(months)), np.nan)
    for col, rep in enumerate(id_reports):
        for row, (orig_path, bk) in enumerate(row_keys):
            if orig_path is None:
                continue
            val = rep.get("scores", {}).get(orig_path, {}).get("buckets", {}).get(bk)
            if val is not None:
                matrix[row, col] = float(val)

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([f"M{m}" for m in months], fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_title("|Pearson corr| – param bucket × month", fontsize=9)
    ax.set_xlabel("Month")
    plt.colorbar(im, ax=ax, label="|corr|")

    # mark threshold line at 0.3
    threshold_val = id_reports[-1].get("threshold", 0.3) if id_reports else 0.3
    # overlay X on identifiable cells
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            v = matrix[row, col]
            if not np.isnan(v) and v >= threshold_val:
                ax.text(col, row, "✓", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")


# ──────────────────────────────────────────────────────────────────────────────
# Panel 4 – Accepted param evolution
# ──────────────────────────────────────────────────────────────────────────────

def plot_param_evolution(records: List[Dict], accepted: List[Dict], ax):
    """
    For each temporal param, plot the bucket values of the accepted config
    at the end of each month using stage_*_best_candidate.json files.
    """
    if not accepted:
        ax.text(0.5, 0.5, "No accepted configs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Accepted param evolution")
        return

    def flatten_buckets(cfg: Dict[str, Any]) -> Dict[str, float]:
        flat = {}
        temporal_paths = [
            "infection_modulation.params.interval_values",
            "mild_detection_modulation.params.interval_values",
            "tracing_modulation.params.interval_values",
        ]
        for path in temporal_paths:
            parts = path.split(".")
            node = cfg
            ok = True
            for p in parts:
                if p in node:
                    node = node[p]
                else:
                    ok = False
                    break
            if not ok or not isinstance(node, list):
                continue
            prefix = parts[0]
            for i, v in enumerate(node):
                flat[f"param_{prefix}_b{i}"] = v
        return flat

    best_per_month = {}
    for acc in accepted:
        m = acc.get("month")
        cfg = acc.get("config", {})
        best_per_month[m] = flatten_buckets(cfg)

    months_seen = sorted(best_per_month.keys())
    param_cols = sorted({k for v in best_per_month.values() for k in v.keys()})
    if not param_cols:
        ax.text(0.5, 0.5, "No temporal params found", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Accepted param evolution")
        return

    prefix_map: Dict[str, List[str]] = {}
    for col in param_cols:
        m = re.match(r"param_([a-z_]+)_b(\d+)$", col)
        if not m:
            continue
        prefix = m.group(1)
        prefix_map.setdefault(prefix, []).append(col)

    cmap = plt.cm.viridis
    for pi, (prefix, cols) in enumerate(sorted(prefix_map.items())):
        cols_sorted = sorted(cols, key=lambda c: int(re.search(r"b(\d+)$", c).group(1)))
        n_buckets = len(cols_sorted)
        for ci, col in enumerate(cols_sorted):
            ys = [_finite(best_per_month[m].get(col)) for m in months_seen if m in best_per_month]
            xs = [m for m in months_seen if m in best_per_month]
            color = cmap(ci / max(n_buckets - 1, 1))
            label = f"{TEMPORAL_SHORT.get(prefix, prefix)[:10]} b{ci}" if ci in (0, n_buckets-1) else None
            ax.plot(xs, ys, "-o", markersize=4, color=color,
                    alpha=0.7, linewidth=1, label=label)

    ax.set_title("Accepted config – temporal param values (by stage)")
    ax.set_xlabel("Stage (fit months)")
    ax.set_ylabel("Param value")
    ax.set_xticks(months_seen)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 5 – Window coverage (Gantt)
# ──────────────────────────────────────────────────────────────────────────────

def plot_window_coverage(records: List[Dict], ax):
    if not records:
        ax.set_title("Window coverage")
        return

    months = sorted({r.get("month", 1) for r in records})
    colours = {m: MONTH_CMAP(i / max(len(months)-1, 1)) for i, m in enumerate(months)}

    plotted_windows = set()
    for r in records:
        m  = r.get("month", 1)
        s  = r.get("start_day")
        e  = r.get("end_day")
        if s is None or e is None:
            continue
        key = (m, s, e)
        if key in plotted_windows:
            continue
        plotted_windows.add(key)
        ax.barh(m, e - s + 1, left=s - 1, height=0.6,
                color=colours[m], alpha=0.5, edgecolor="k", linewidth=0.4)

    ax.set_yticks(months)
    ax.set_yticklabels([f"Month {m}" for m in months])
    ax.set_xlabel("Simulation day")
    ax.set_title("Search window coverage")
    ax.grid(True, axis="x", alpha=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Combined figure
# ──────────────────────────────────────────────────────────────────────────────

def make_figure(output_dir: Path, show: bool = False):
    records    = load_progress(output_dir)
    id_reports = load_identifiability(output_dir)
    accepted   = load_accepted_months(output_dir)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Checkpoint delta search — {output_dir.name}", fontsize=12, y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    ax_combined  = fig.add_subplot(gs[0, 0])
    ax_metrics   = fig.add_subplot(gs[0, 1])
    ax_ident     = fig.add_subplot(gs[1, :])
    ax_param_evo = fig.add_subplot(gs[2, 0])
    ax_windows   = fig.add_subplot(gs[2, 1])

    plot_rmse_timeline(records, ax_combined, ax_metrics)
    plot_identifiability(id_reports, ax_ident)
    plot_param_evolution(records, accepted, ax_param_evo)
    plot_window_coverage(records, ax_windows)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    combined_path = plots_dir / "search_overview.png"
    fig.savefig(combined_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {combined_path}")

    # also save individual panels
    panel_specs = [
        ("rmse_timeline",    [ax_combined, ax_metrics]),
        ("identifiability",  [ax_ident]),
        ("param_evolution",  [ax_param_evo]),
        ("window_coverage",  [ax_windows]),
    ]
    for name, axes in panel_specs:
        extent = _full_extent(axes, fig)
        panel_path = plots_dir / f"{name}.png"
        fig.savefig(panel_path, dpi=130, bbox_inches=extent)
        print(f"Saved: {panel_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)


def _full_extent(axes, fig):
    """Return a tight bounding box (in figure coordinates) covering all given axes."""
    from matplotlib.transforms import Bbox
    items = []
    for ax in axes:
        items.append(ax.get_tightbbox(fig.canvas.get_renderer()))
        for child in ax.get_children():
            try:
                bb = child.get_window_extent(fig.canvas.get_renderer())
                items.append(bb)
            except Exception:
                pass
    total = Bbox.union([b for b in items if b is not None and b.width > 0])
    return total.transformed(fig.dpi_scale_trans.inverted())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot checkpoint delta search results")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory written by run_checkpoint_search.jl (contains rmse_progress.json)")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive matplotlib window after saving")
    args = parser.parse_args()
    make_figure(args.output_dir.resolve(), show=args.show)


if __name__ == "__main__":
    main()
