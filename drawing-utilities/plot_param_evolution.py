#!/usr/bin/env python3
"""Plot evolution of all temporal param buckets across every config.json in the search."""

import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SEARCH_DIR = "/Users/marcinbodych/Workspace/saxocov/experiment-wcss/LLM-IDEAS/manager/runs/search_local"
MODULATIONS = [
    ("infection_modulation",       "Infection modulation"),
    ("mild_detection_modulation",  "Detection modulation"),
    ("tracing_modulation",         "Tracing modulation"),
]

def sort_key(path):
    nums = re.findall(r'\d+', path)
    return [int(n) for n in nums]

def load_all_configs(search_dir):
    files = sorted(
        glob.glob(os.path.join(search_dir, "**/config.json"), recursive=True),
        key=sort_key
    )
    records = []
    for f in files:
        try:
            cfg = json.load(open(f))
            parts = f.replace(os.sep, "/").split("/")
            # extract month/week/iter/candidate from path
            month = next((int(re.search(r'\d+', p).group()) for p in parts if p.startswith("month_")), 0)
            week  = next((int(re.search(r'\d+', p).group()) for p in parts if p.startswith("week_")),  0)
            label = next((p for p in parts if p.startswith("iter_")), "")
            records.append({"path": f, "config": cfg, "month": month, "week": week, "label": label})
        except Exception:
            pass
    return records

records = load_all_configs(SEARCH_DIR)
n = len(records)
print(f"Loaded {n} configs")

# colour by month
month_colors = {1: "#2196F3", 2: "#E91E63", 3: "#9C27B0", 4: "#009688"}

for mod_key, mod_title in MODULATIONS:
    # collect all bucket series: shape (n_configs, n_buckets)
    try:
        n_buckets = len(records[0]["config"][mod_key]["params"]["interval_values"])
        interval_times = records[0]["config"][mod_key]["params"]["interval_times"]
    except (KeyError, IndexError):
        continue

    values = np.full((n, n_buckets), np.nan)
    for i, rec in enumerate(records):
        try:
            vals = rec["config"][mod_key]["params"]["interval_values"]
            values[i] = vals
        except (KeyError, TypeError):
            pass

    # one subplot per bucket
    cols = 5
    rows = (n_buckets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8), sharey=True)
    fig.suptitle(f"{mod_title} — all {n_buckets} buckets over {n} configs", fontsize=12)
    axes = axes.flatten()

    x = np.arange(n)
    months = np.array([r["month"] for r in records])
    # month boundary positions
    boundaries = [0]
    for m in [1, 2, 3, 4]:
        idx = np.where(months == m)[0]
        if len(idx):
            boundaries.append(idx[-1] + 1)

    for b in range(n_buckets):
        ax = axes[b]
        y = values[:, b]

        # shade months
        shade_colors = ["#E3F2FD", "#FCE4EC", "#F3E5F5", "#E8F5E9"]
        for mi, m in enumerate([1, 2, 3, 4]):
            idx = np.where(months == m)[0]
            if len(idx):
                ax.axvspan(idx[0], idx[-1] + 1, alpha=0.18, color=shade_colors[mi], zorder=0)

        # plot each month segment separately for colour
        for m, col in month_colors.items():
            idx = np.where(months == m)[0]
            if len(idx):
                ax.plot(idx, y[idx], "o-", color=col, ms=2, lw=1.2, alpha=0.8)

        # mark month boundaries
        for bnd in boundaries[1:-1]:
            ax.axvline(bnd, color="#999", lw=0.7, ls=":")

        # bucket label: days covered
        t_start = 0 if b == 0 else interval_times[b - 1]
        t_end   = interval_times[b] if b < len(interval_times) else "∞"
        ax.set_title(f"b{b}  [{t_start}–{t_end}d]", fontsize=8)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("config idx", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # hide unused subplots
    for b in range(n_buckets, len(axes)):
        axes[b].set_visible(False)

    # legend
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=f"Month {m}")
               for m, c in month_colors.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=9, ncol=4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = os.path.join(SEARCH_DIR, "plots", f"param_evolution_{mod_key}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
