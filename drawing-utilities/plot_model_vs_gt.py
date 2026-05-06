#!/usr/bin/env python3
"""Plot model daily output vs ground truth for each metric over the full 4-month period."""

import argparse, os, csv, glob, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py

def read_gt_csv(path):
    values = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            try:
                values.append(float(row[1]))
            except (IndexError, ValueError):
                pass
    return np.array(values)

def read_jld2_metric(path, metric):
    """Read a metric from a JLD2 (HDF5) daily output file, averaged over trajectories."""
    with h5py.File(path, "r") as f:
        arrays = []
        for traj_key in f.keys():
            grp = f[traj_key]
            if metric in grp:
                arrays.append(np.array(grp[metric], dtype=float))
        if not arrays:
            return None
        max_len = max(len(a) for a in arrays)
        padded = np.zeros((len(arrays), max_len))
        for i, a in enumerate(arrays):
            padded[i, :len(a)] = a
        return padded.mean(axis=0)

def rolling_avg(arr, window=7):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = arr[start:i+1].mean()
    return result

def find_best_daily(search_dir, month, week="04"):
    """Find daily.jld2 via accepted_month_XX.json checkpoint; fallback to week search."""
    accepted = os.path.join(search_dir, f"accepted_month_{month:02d}.json")
    if os.path.isfile(accepted):
        d = json.load(open(accepted))
        ckpt = d.get("checkpoint_path")
        if ckpt:
            cand = os.path.join(os.path.dirname(ckpt), "daily.jld2")
            if os.path.isfile(cand):
                return cand
    pattern = os.path.join(search_dir, f"month_{month:02d}", f"week_{week}", "*", "daily.jld2")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None
    return files[0] if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", default=None,
        help="Path to search output containing accepted_month_XX.json and daily.jld2")
    parser.add_argument("--gt-dir", default=None,
        help="Path containing daily_detections.csv, daily_deaths.csv, daily_hospitalizations.csv")
    parser.add_argument("--month", type=int, default=None,
                        help="Which month's best daily to use (default: highest completed)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # defaults
    if args.search_dir is None:
        args.search_dir = "/Users/marcinbodych/Workspace/saxocov/experiment-wcss/LLM-IDEAS/manager/runs/search_local"
    if args.gt_dir is None:
        args.gt_dir = os.path.normpath(os.path.join(args.search_dir, "..", "..", "gt"))

    gt_files = {
        "daily_detections":       os.path.join(args.gt_dir, "daily_detections.csv"),
        "daily_deaths":           os.path.join(args.gt_dir, "daily_deaths.csv"),
        "daily_hospitalizations": os.path.join(args.gt_dir, "daily_hospitalizations.csv"),
    }
    titles = {
        "daily_detections":       "Daily Detections",
        "daily_deaths":           "Daily Deaths",
        "daily_hospitalizations": "Daily Hospitalizations",
    }
    colors = {
        "daily_detections":       "#2196F3",
        "daily_deaths":           "#F44336",
        "daily_hospitalizations": "#FF9800",
    }

    # find the highest completed month if not specified
    if args.month is not None:
        target_month = args.month
    else:
        import glob as _glob
        completed = sorted(_glob.glob(os.path.join(args.search_dir, "accepted_month_*.json")))
        if not completed:
            print("No accepted_month_*.json found")
            return
        target_month = int(os.path.basename(completed[-1]).split("_")[2].split(".")[0])

    best_file = find_best_daily(args.search_dir, month=target_month, week="04")
    if best_file is None:
        best_file = find_best_daily(args.search_dir, month=target_month, week="01")
    if best_file is None:
        print(f"No daily.jld2 found for month {target_month}")
        return
    print(f"Using month {target_month}: {best_file}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Model vs Ground Truth — through month {target_month}", fontsize=13)

    total_days = 112
    days = np.arange(1, total_days + 1)
    month_boundaries = [1, 29, 57, 85, 113]
    month_labels = ["Month 1\n(Sep)", "Month 2\n(Oct)", "Month 3\n(Nov)", "Month 4\n(Dec)"]
    month_colors = ["#E3F2FD", "#FCE4EC", "#F3E5F5", "#E8F5E9"]

    for ax, metric in zip(axes, gt_files):
        # ground truth
        gt = read_gt_csv(gt_files[metric])
        gt_days = np.arange(1, len(gt) + 1)

        # model prediction
        pred = read_jld2_metric(best_file, metric)

        # shade months
        for i, (lo, hi) in enumerate(zip(month_boundaries[:-1], month_boundaries[1:])):
            ax.axvspan(lo, hi, alpha=0.15, color=month_colors[i], zorder=0)
            if metric == "daily_detections":
                ax.text((lo + hi) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                        month_labels[i], ha="center", va="top", fontsize=8, color="#555")

        # plot GT (raw + 7-day rolling)
        ax.bar(gt_days[:total_days], gt[:total_days], color=colors[metric],
               alpha=0.25, width=1.0, label="GT raw")
        ax.plot(gt_days[:total_days], rolling_avg(gt[:total_days]),
                color=colors[metric], lw=2, label="GT 7-day avg")

        # plot model
        if pred is not None:
            pred_days = np.arange(1, len(pred) + 1)
            ax.bar(pred_days[:total_days], pred[:total_days], color="#607D8B",
                   alpha=0.2, width=1.0, label="Model raw")
            ax.plot(pred_days[:total_days], rolling_avg(pred[:total_days]),
                    color="#37474F", lw=2, ls="--", label="Model 7-day avg")

        # month boundary lines
        for b in month_boundaries[1:-1]:
            ax.axvline(b, color="#999", lw=0.8, ls=":")

        ax.set_ylabel(titles[metric], fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(1, total_days)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Simulation day (day 1 = 2020-09-03)", fontsize=10)

    plt.tight_layout()
    out = args.out or os.path.join(args.search_dir, "plots", "model_vs_gt.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
